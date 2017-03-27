from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json


def get_config(dataset, model):
  # Use one of the pre-set config.
  if model == "resnet-32":
    config = ResNet32Config()
  elif model == "resnet-110":
    config = ResNet110Config()
  elif model == "resnet-164":
    config = ResNet164Config()
  else:
    raise Exception("Unknown model \"{}\"".format(model))
  if dataset == "cifar-10":
    config.num_classes = 10
  elif dataset == "cifar-100":
    config.num_classes = 100
  else:
    raise Exception("Unknown dataset")
  return config


class ResNet32Config(object):

  def __init__(self):
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
    self.filters = [16, 16, 32, 64]
    self.wd = 2e-4
    # self.relu_leakiness = 0.1   # Original TF model has leaky relu.
    self.relu_leakiness = 0.0
    self.optimizer = "mom"
    self.max_train_iter = 80000
    self.lr_scheduler_type = 'fixed'
    self.lr_decay_steps = [40000, 60000]
    self.lr_list = [1e-2, 1e-3]
    self.model = "resnet-32"
    self.disp_iter = 100
    self.save_iter = 10000
    self.valid_iter = 1000
    self.norm_field = None
    self.sigma_init = 1e-2
    self.learn_sigma = False
    self.norm_affine = False
    self.stagewise_norm = False
    self.l1_reg = 0.0
    self.prefetch = True
    self.data_aug = True
    self.whiten = False  # Original TF has whiten.
    self.div255 = True
    self.seed = 0

  def to_json(self):
    return json.dumps(self, default=lambda o: o.__dict__)

  @classmethod
  def from_json(cls, s):
    dic = json.loads(s)
    config = cls()
    config.__dict__ = dic
    return config


class ResNet110Config(ResNet32Config):

  def __init__(self):
    super(ResNet110Config, self).__init__()
    self.num_residual_units = [18, 18, 18]  # ResNet-110
    self.model = "resnet-110"


class ResNet164Config(ResNet32Config):

  def __init__(self):
    super(ResNet164Config, self).__init__()
    self.num_residual_units = [18, 18, 18]  # ResNet-164
    self.use_bottleneck = True
    self.model = "resnet-164"


def get_config_from_json(path):
  return ResNet32Config.from_json(open(path, "r").read())
