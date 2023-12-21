import logging
import os
import time
import torch
import torch.nn as nn
from loss.triplet_loss import euclidean_dist, hard_example_mining
from model.make_model import make_model
from processor.inf_processor import do_inference
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from data.build_DG_dataloader import build_reid_test_loader, build_reid_train_loader
from torch.utils.tensorboard import SummaryWriter

def mix_vit_do_train_with_amp(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank,
             train_dir=None,
             num_pids=None,
             **kwargs):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid.train")
    logger.info('start training')
    log_path = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)
    tb_path = os.path.join(cfg.TB_LOG_ROOT, cfg.LOG_NAME)
    tbWriter = SummaryWriter(tb_path)
    print("saving tblog to {}".format(tb_path))
    
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    loss_id_meter = AverageMeter()
    loss_id_distinct_meter = AverageMeter()
    loss_tri_meter = AverageMeter()
    loss_center_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler(init_scale=512)
    bs = cfg.SOLVER.IMS_PER_BATCH
    num_ins = cfg.DATALOADER.NUM_INSTANCE
    classes = len(train_loader.dataset.pids)
    center_weight = cfg.SOLVER.CENTER_LOSS_WEIGHT

    best = 0.0
    best_index = 1
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        loss_id_meter.reset()
        loss_id_distinct_meter.reset()
        loss_tri_meter.reset()
        loss_center_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()            
        
        for n_iter, informations in enumerate(train_loader):
            img = informations['images']
            vid = informations['targets']
            target_cam = informations['camid']
            ori_label = informations['ori_label']
            t_domains = informations['others']['domains']

            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            ori_label = ori_label.to(device)
            t_domains = t_domains.to(device)

            targets = torch.zeros((bs, classes)).scatter_(1, target.unsqueeze(1).data.cpu(), 1).to(device)
            model.to(device)
            with amp.autocast(enabled=True):
                score, feat, target, score_ = model(img, target, t_domains)

                # #### id loss
                # log_probs = nn.LogSoftmax(dim=1)(score)
                # targets = 0.9 * targets + 0.1 / classes # label smooth
                # loss_id = (- targets * log_probs).mean(0).sum()
                loss_id = torch.tensor(0.0,device=device) ####### for test

                #### id loss for each domain
                loss_id_distinct = torch.tensor(0.0, device=device)
                for i,s in enumerate(score_):
                    if s is None: continue
                    idx = torch.nonzero(t_domains==i).squeeze(1)
                    log_probs = nn.LogSoftmax(1)(s)
                    label = torch.zeros((len(idx), num_pids[i])).scatter_(1, ori_label[idx].unsqueeze(1).data.cpu(), 1).to(device)
                    label = 0.9 * label + 0.1 / num_pids[i] # label smooth
                    loss_id_distinct += (- label * log_probs).mean(0).sum()


                dist_mat = euclidean_dist(feat, feat)
                dist_ap, dist_an = hard_example_mining(dist_mat, target)
                y = dist_an.new().resize_as_(dist_an).fill_(1)
                loss_tri = nn.SoftMarginLoss()(dist_an - dist_ap, y)
                # loss_tri = torch.tensor(0.0, device=device)

                #### center loss
                if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                    loss_center = center_criterion(feat, target)
                else:
                    loss_center = torch.tensor(0.0, device=device)

                loss = loss_id + loss_tri + loss_id_distinct\
                    + center_weight * loss_center\

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), bs)
            loss_id_meter.update(loss_id.item(), bs)
            loss_id_distinct_meter.update(loss_id_distinct.item(), bs)
            loss_tri_meter.update(loss_tri.item(), bs)
            loss_center_meter.update(center_weight*loss_center.item(), bs)
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, id:{:.3f}, id_dis:{:.3f}, tri:{:.3f}, cen:{:.3f}  Acc: {:.3f}, Base Lr: {:.2e}"
                .format(epoch, n_iter+1, len(train_loader),
                loss_meter.avg, 
                loss_id_meter.avg, loss_id_distinct_meter.avg, loss_tri_meter.avg,
                loss_center_meter.avg,
                acc_meter.avg, scheduler._get_lr(epoch)[0]))
                tbWriter.add_scalar('train/loss', loss_meter.avg, n_iter+1+(epoch-1)*len(train_loader))
                tbWriter.add_scalar('train/acc', acc_meter.avg, n_iter+1+(epoch-1)*len(train_loader))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]".format(epoch, time_per_batch, cfg.SOLVER.IMS_PER_BATCH / time_per_batch))

        log_path = os.path.join(cfg.LOG_ROOT, cfg.LOG_NAME)

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                cmc, mAP = do_inference(cfg, model, val_loader, num_query)
                tbWriter.add_scalar('val/Rank@1', cmc[0], epoch)
                tbWriter.add_scalar('val/mAP', mAP, epoch)
                torch.cuda.empty_cache()
        if epoch % eval_period == 0:
            if best < mAP + cmc[0]:
                best = mAP + cmc[0]
                best_index = epoch
                logger.info("=====best epoch: {}=====".format(best_index))
                if cfg.MODEL.DIST_TRAIN:
                    if dist.get_rank() == 0:
                        torch.save(model.state_dict(),
                                os.path.join(log_path, cfg.MODEL.NAME + '_best.pth'))
                else:
                    torch.save(model.state_dict(),
                            os.path.join(log_path, cfg.MODEL.NAME + '_best.pth'))
        torch.cuda.empty_cache()

    # final evaluation
    load_path = os.path.join(log_path, cfg.MODEL.NAME + '_best.pth')
    eval_model = make_model(cfg, modelname=cfg.MODEL.NAME, num_class=0)
    eval_model.load_param(load_path)
    logger.info('load weights from best.pth')
    if 'DG' in cfg.DATASETS.TEST[0]:
        do_inference_multi_targets(cfg, model, logger)
    else:
        for testname in cfg.DATASETS.TEST:
            val_loader, num_query = build_reid_test_loader(cfg, testname)
            do_inference(cfg, model, val_loader, num_query)
