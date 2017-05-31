from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from resnet.configs.config_factory import RegisterConfig
from resnet.configs.cifar_configs import ResNet32Cifar10Config


class ResNetTestConfig(ResNet32Cifar10Config):

  def __init__(self):
    super(ResNetTestConfig, self).__init__()
    self.batch_size = 10
    self.num_residual_units = [2, 2, 2]
    self.filters = [2, 2, 4, 8]


@RegisterConfig("test", "resnet-test")
def get_resnet_test():
  return ResNetTestConfig()


class ResNetBottleneckTestConfig(ResNetTestConfig):

  def __init__(self):
    super(ResNetBottleneckTestConfig, self).__init__()
    self.use_bottlneck = True


@RegisterConfig("test", "resnet-btl-test")
def get_resnet_test():
  return ResNetTestConfig()
