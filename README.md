# resnet
Standard ResNet training on image classification benchmarks. Modified from the original tensorflow version.

## Installation
Custom paths first in `setup.sh` (data folder, model save folder, etc.).
```bash
git clone --recursive git://github.com/renmengye/resnet.git
cd resnet
./setup.sh
```

## CIFAR-10/100
```bash
./run_cifar_exp.py --dataset cifar-10 --model resnet-32
```

## ImageNet
```
./run_imagenet_exp.py --model resnet-50
./run_imagenet_eval.py --id [EXPERIMENT ID]
```

## ImageNet on Slurm
SSH into the slurm manager node first, and then launch jobs there.
```
# Launch a recurring training job, 30K steps per job, for total 600K stes.
./run_imagenet_exp_sched.py --model resnet-50 --max_num_steps 30000 --max_max_steps 600000

# Launch a recurring evaluation job every 2 hours.
./run_imagenet_eval_sched.py --id [EXPERIMENT ID] --min_interval 7200
```

## Provided Model Configs
See `resnet/configs/cifar_exp_config.py` and `resnet/configs/imagenet_exp_config.py`
