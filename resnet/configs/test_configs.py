from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from resnet.configs.config_factory import RegisterConfig
from resnet.configs.cifar_configs import (ResNet32Cifar10Config,
                                          RevResNet38Cifar10Config)


class ResNetTestConfig(ResNet32Cifar10Config):

  def __init__(self):
    super(ResNetTestConfig, self).__init__()
    self.batch_size = 10
    self.num_residual_units = [2, 2, 2]
    self.filters = [2, 2, 4, 8]


@RegisterConfig("test", "resnet-test")
def get_resnet_test():
  return ResNetTestConfig()


class RevResNetTestConfig(RevResNet38Cifar10Config):

  def __init__(self):
    super(RevResNetTestConfig, self).__init__()
    self.batch_size = 10
    self.num_residual_units = [2, 2]
    self.filters = [16, 16, 32]
    self.height = 8
    self.width = 8
    self.model_class = "revresnet"


@RegisterConfig("test", "revresnet-test")
def get_revresnet_test():
  return RevResNetTestConfig()


class RevResNetBottleneckTestConfig(RevResNet38Cifar10Config):

  def __init__(self):
    super(RevResNetBottleneckTestConfig, self).__init__()
    self.batch_size = 10
    self.num_residual_units = [2, 2]
    self.filters = [16, 16, 32]
    self.height = 8
    self.width = 8
    self.model_class = "revresnet"
    self.use_bottleneck = True


@RegisterConfig("test", "revresnet-btl-test")
def get_revresnet_btl_test():
  return RevResNetBottleneckTestConfig()


class RevResNetV2TestConfig(RevResNetTestConfig):

  def __init__(self):
    super(RevResNetV2TestConfig, self).__init__()
    self.model_class = "revresnet-v2"


@RegisterConfig("test", "revresnet-v2-test")
def get_revresnet_v2_test():
  return RevResNetV2TestConfig()


class RevResNetBottleneckV2TestConfig(RevResNetBottleneckTestConfig):

  def __init__(self):
    super(RevResNetBottleneckV2TestConfig, self).__init__()
    self.model_class = "revresnet-v2"


@RegisterConfig("test", "revresnet-btl-v2-test")
def get_revresnet_btl_v2_test():
  return RevResNetBottleneckV2TestConfig()