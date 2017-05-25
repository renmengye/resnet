from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from resnet.configs.config_factory import RegisterConfig


class ResNet50Config(object):

  def __init__(self):
    super(ResNet50Config, self).__init__()
    self.height = 224
    self.width = 224
    self.num_channel = 3
    self.num_residual_units = [3, 4, 6, 3]  # ResNet-50
    self.strides = [1, 2, 2, 2]
    self.activate_before_residual = [True, False, False, False]
    self.init_stride = 2
    self.init_max_pool = True
    self.init_filter = 7
    self.use_bottleneck = True
    self.relu_leakiness = False
    self.filters = [64, 64, 128, 256, 512]
    self.wd = 1e-4
    self.momentum = 0.9
    self.base_learn_rate = 1e-1
    self.max_train_iter = 600000
    self.lr_scheduler = "fixed"
    self.lr_decay_steps = [160000, 320000, 480000]
    self.lr_list = [1e-2, 1e-3, 1e-4]
    self.name = "resnet-50"
    self.model_class = "resnet"
    self.disp_iter = 10
    self.save_iter = 5000
    self.trainval_iter = 1000
    self.valid_iter = 5000
    self.valid_batch_size = 64  ### Use this if necessary.
    self.div255 = True
    self.run_validation = False
    self.num_classes = 1000
    self.batch_size = 256
    self.preprocessor = "inception"  # VGG or Inception.
    self.seed = 1234
    self.optimizer = "mom"
    self.filter_initialization = "normal"


@RegisterConfig("imagenet", "resnet-50")
def get_resnet_50_config():
  return ResNet50Config()


class ResNet50GPUConfig(ResNet50Config):

  def __init__(self):
    super(ResNet50GPUConfig, self).__init__()
    self.model_class = "resnet-gpu"
    self.batch_size = 256
    self.name = "resnet-50-gpu"


@RegisterConfig("imagenet", "resnet-50-gpu")
def get_resnet_50_gpu_config():
  return ResNet50GPUConfig()


class ResNet101Config(ResNet50Config):

  def __init__(self):
    super(ResNet101Config, self).__init__()
    self.num_residual_units = [3, 4, 23, 3]  # ResNet-101
    self.batch_size = 256
    self.name = "resnet-101"


@RegisterConfig("imagenet", "resnet-101")
def get_resnet_101_config():
  return ResNet101Config()


class ResNet152Config(ResNet50Config):

  def __init__(self):
    super(ResNet152Config, self).__init__()
    self.num_residual_units = [3, 8, 36, 3]  # ResNet-152
    self.batch_size = 256
    self.name = "resnet-152"


@RegisterConfig("imagenet", "resnet-152")
def get_resnet_152_config():
  return ResNet152Config()
