import os
import torch
import numpy as np
import sys
import collections.abc as container_abcs

from data.samplers.triplet_sampler import SHS
from loss.center_loss import CenterLoss

# from data.transforms.build import build_transform_local
# from torch._six import container_abcs, string_classes, int_classes
int_classes = int
string_classes = str
from torch.utils.data import DataLoader
from utils import comm
import random

from . import samplers
from .common import CommDataset
from .datasets import DATASET_REGISTRY
from .transforms import build_transforms

# export REID_DATASETS="/xxx/xxx"
# _root = os.getenv("REID_DATASETS", "/home/liyuke/data")


def build_reid_train_loader(cfg):
    # gettrace = getattr(sys, 'gettrace', None)
    # if gettrace():
    #     print('*'*100)
    #     print('Hmm, Big Debugger is watching me')
    #     print('*'*100)
    #     num_workers = 0
    # else:
    num_workers = cfg.DATALOADER.NUM_WORKERS

    train_transforms = build_transforms(cfg, is_train=True, is_fake=False)
    train_items = list()
    domain_idx = 0
    camera_all = list()

    # load datasets
    train_pids = []
    domain_names = []
    _root = cfg.DATASETS.ROOT_DIR
    for d in cfg.DATASETS.TRAIN:
        if d == 'CUHK03_NP':
            dataset = DATASET_REGISTRY.get('CUHK03')(root=_root, cuhk03_labeled=False)
        else:
            dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL)
        if comm.is_main_process():
            dataset.show_train()
        if len(dataset.train[0]) < 4:
            for i, x in enumerate(dataset.train):
                add_info = {}  # dictionary

                if cfg.DATALOADER.CAMERA_TO_DOMAIN and len(cfg.DATASETS.TRAIN) == 1:
                    add_info['domains'] = dataset.train[i][2]
                    camera_all.append(dataset.train[i][2])
                else:
                    add_info['domains'] = int(domain_idx)
                dataset.train[i] = list(dataset.train[i])
                dataset.train[i].append(add_info)
                dataset.train[i] = tuple(dataset.train[i])
        domain_idx += 1
        domain_names.append(dataset.dataset_name)
        train_items.extend(dataset.train)
        train_pids.append(dataset.get_num_pids(dataset.train))

    train_set = CommDataset(cfg, train_items, train_transforms, relabel=True, domain_names=domain_names)

    if len(cfg.DATASETS.TRAIN) == 1 and cfg.DATALOADER.CAMERA_TO_DOMAIN:
        num_domains = dataset.num_train_cams
    else:
        num_domains = len(cfg.DATASETS.TRAIN)
    cfg.defrost()
    cfg.DATASETS.NUM_DOMAINS = num_domains
    cfg.freeze()

    train_loader, centers, model = make_sampler(
        train_set=train_set,
        num_batch=cfg.SOLVER.IMS_PER_BATCH,
        num_instance=cfg.DATALOADER.NUM_INSTANCE,
        num_workers=num_workers,
        mini_batch_size=cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size(),
        drop_last=cfg.DATALOADER.DROP_LAST,
        flag1=cfg.DATALOADER.NAIVE_WAY,
        flag2=cfg.DATALOADER.DELETE_REM,
        train_pids=train_pids,
        cfg = cfg)


    return train_loader, num_domains, train_pids, centers, model


def build_reid_test_loader(cfg, dataset_name, opt=None, flag_test=True, shuffle=False, only_gallery=False, only_query=False, eval_time=False, bs=None):
    test_transforms = build_transforms(cfg, is_train=False)
    _root = cfg.DATASETS.ROOT_DIR
    if opt is None:
        dataset = DATASET_REGISTRY.get(dataset_name)(root=_root)
        if comm.is_main_process():
            if flag_test:
                dataset.show_test()
            else:
                dataset.show_train()
    else:
        dataset = DATASET_REGISTRY.get(dataset_name)(root=[_root, opt])
    if flag_test:
        if only_gallery:
            test_items = dataset.gallery
        elif only_query:
            test_set = CommDataset(cfg, [random.choice(dataset.query)], test_transforms, relabel=False)
            return test_set
        else:
            test_items = dataset.query + dataset.gallery
        if shuffle: # only for visualization
            random.shuffle(test_items)
    else:
        test_items = dataset.train

    test_set = CommDataset(cfg, test_items, test_transforms, relabel=False)

    batch_size = bs if bs is not None else cfg.TEST.IMS_PER_BATCH
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, batch_size, False)

    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        num_workers = 0
    else:
        num_workers = cfg.DATALOADER.NUM_WORKERS

    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=num_workers,  # save some memory
        collate_fn=fast_batch_collator)
    return test_loader, len(dataset.query)


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out

    elif isinstance(elem, container_abcs.Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs

def make_sampler(train_set, num_batch, num_instance, num_workers,
                 mini_batch_size, drop_last=True, flag1=True, flag2=True, seed=None, train_pids=None, cfg=None):

    #### center loss initiation
    num_classes = 0
    if isinstance(train_pids, list):
        for i in train_pids:
            num_classes += i
    else:
        num_classes = train_pids
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=cfg.MODEL.DIM)
    from model import make_model
    model = make_model(cfg, modelname=cfg.MODEL.NAME, num_class=num_classes, num_class_domain_wise=train_pids)
    test_transforms = build_transforms(cfg, is_train=False)
    if cfg.SOLVER.RESUME:
        model.load_param(cfg.SOLVER.RESUME_PATH)

    if cfg.DATALOADER.SAMPLER == 'SHS':
        data_sampler = SHS(cfg=cfg,centers=center_criterion.centers,
                                            train_set=train_set,
                                            batch_size=mini_batch_size, num_pids=train_pids,model=model,
                                            transform=test_transforms)
    elif cfg.DATALOADER.SAMPLER == 'single_domain':
        data_sampler = samplers.DomainIdentitySampler(train_set.img_items,
                                                      mini_batch_size, num_instance,train_pids)
    elif flag1:
        data_sampler = samplers.RandomIdentitySampler(train_set.img_items,
                                                      mini_batch_size, num_instance)
    else:
        data_sampler = samplers.DomainSuffleSampler(train_set.img_items,
                                                     num_batch, num_instance, flag2, seed, cfg)
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, drop_last)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
    )
    return train_loader, center_criterion, model