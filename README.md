# Style-Controllable-Generalized-Person-Re-identification
The official repo for Style Controllable Generalized Person Re-identification [ACM MM 2023]

Here are some instructions to run our code.
Our code is based on [TransReID](https://github.com/damo-cv/TransReID), thanks for their excellent work.

## 1. clone this repo
```
git clone https://github.com/liyuke65535/Style-Controllable-Generalized-Person-Re-identification.git
```

## 2. prepare your environment
```
conda create -n screid python==3.10
conda activate screid
bash enviroments.sh
```

## 3. prepare pretrained model (ViT-B) and datasets
You can download it from huggingface, rwightman or else where.
For example, pretrained model is avaliable at [ViT-B](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth).

As for datasets, follow the instructions in [MetaBIN](https://github.com/bismex/MetaBIN#8-datasets).

## 4. alter the config file
```
# alter the model path and dataset paths of the config file
vim ./config/SHS_DSM_vit_b.yml
```

## 5. train a model
```
bash run.sh
```

## 6. evaluation only
```
# alter the trained path in config
vim ./config/SHS_DSM_vit_b.yml

# evaluation
python test.py --config ./config/SHS_DSM_vit_b.yml
```
## Citation
```
