from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from resnet.configs.config_factory import RegisterConfig


class ResNet32Cifar10Config(object):

  def __init__(self):
    super(ResNet32Cifar10Config, self).__init__()
    self.batch_size = 100
    self.height = 32
    self.width = 32
    self.num_channel = 3
    self.min_lrn_rate = 0.0001
    self.base_learn_rate = 1e-1
    self.num_residual_units = [5, 5, 5]  # ResNet-32
    self.seed = 1234
    self.strides = [1, 2, 2]
    self.activate_before_residual = [True, False, False]
    self.init_stride = 1
    self.init_max_pool = False
    self.init_filter = 3
    self.use_bottleneck = False
    self.relu_leakiness = False
    self.filters = [16, 16, 32, 64]
    self.wd = 2e-4
    self.optimizer = "mom"
    self.max_train_iter = 80000
    self.lr_decay_steps = [40000, 60000]
    self.lr_scheduler_type = "fixed"
    self.lr_list = [1e-2, 1e-3]
    self.momentum = 0.9
    self.name = "resnet-32"
    self.model_class = "resnet"
    self.filter_initialization = "normal"
    self.disp_iter = 100
    self.save_iter = 10000
    self.valid_iter = 1000
    self.prefetch = True
    self.data_aug = True
    self.whiten = False  # Original TF has whiten.
    self.div255 = True
    self.seed = 0
    self.num_classes = 10


@RegisterConfig("cifar-10", "resnet-32")
def get_cifar10_resnet32():
  return ResNet32Cifar10Config()


class ResNet32Cifar10GPUConfig(ResNet32Cifar10Config):

  def __init__(self):
    super(ResNet32Cifar10GPUConfig, self).__init__()
    self.model_class = "resnet-gpu"


@RegisterConfig("cifar-10", "resnet-32-gpu")
def get_cifar10_resnet32_gpu():
  return ResNet32Cifar10GPUConfig()


class ResNet32Cifar100Config(ResNet32Cifar10Config):

  def __init__(self):
    super(ResNet32Cifar100Config, self).__init__()
    self.num_classes = 100


@RegisterConfig("cifar-100", "resnet-32")
def get_cifar100_resnet32():
  return ResNet32Cifar100Config()


class ResNet110Cifar10Config(ResNet32Cifar10Config):

  def __init__(self):
    super(ResNet110Cifar10Config, self).__init__()
    self.num_residual_units = [18, 18, 18]  # ResNet-110
    self.name = "resnet-110"


@RegisterConfig("cifar-10", "resnet-110")
def get_cifar10_resnet110():
  return ResNet110Cifar10Config()


class ResNet110Cifar100Config(ResNet110Cifar10Config):

  def __init__(self):
    super(ResNet110Cifar100Config, self).__init__()
    self.num_classes = 100


@RegisterConfig("cifar-100", "resnet-110")
def get_cifar100_resnet110():
  return ResNet110Cifar100Config()


class ResNet164Cifar10Config(ResNet32Cifar10Config):

  def __init__(self):
    super(ResNet164Cifar10Config, self).__init__()
    self.num_residual_units = [18, 18, 18]  # ResNet-164
    self.use_bottleneck = True
    self.name = "resnet-164"


@RegisterConfig("cifar-10", "resnet-164")
def get_cifar10_resnet164():
  return ResNet164Cifar10Config()


class ResNet164Cifar100Config(ResNet164Cifar10Config):

  def __init__(self):
    super(ResNet164Cifar100Config, self).__init__()
    self.num_classes = 100


@RegisterConfig("cifar-100", "resnet-164")
def get_cifar100_resnet164():
  return ResNet164Cifar100Config()
