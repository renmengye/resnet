# resnet
Modified from the original tensorflow version.

## Installation
Custom paths first in `setup.sh` (data folder, model save folder, etc.).
```bash
git clone --recursive git://github.com/renmengye/resnet.git
cd resnet
./setup.sh
```

## CIFAR
```bash
./run_cifar_exp.py --dataset cifar-10 --model resnet-32
```

## ImageNet
```
./run_imagenet_exp.py --model resnet-50
./run_imagenet_eval.py --id [EXPERIMENT ID]
```

## Provided Model Configs
See `resnet/configs/cifar_exp_config.py` and `resnet/configs/imagenet_exp_config.py`
