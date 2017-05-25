"""Same as ResNetModel, except that all weight variables (not computation!)
stay on GPU.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import tensorflow as tf

from resnet.models.nnlib import weight_variable
from resnet.models.model_factory import RegisterModel
from resnet.models.resnet_model import ResNetModel


class ResNetModelGPU(ResNetModel):

  def _weight_variable(self,
                       shape,
                       init_method=None,
                       dtype=tf.float32,
                       init_param=None,
                       wd=None,
                       name=None,
                       trainable=True,
                       seed=0):
    """Wrapper to declare variables, on GPU."""
    return weight_variable(
        shape,
        init_method=init_method,
        dtype=dtype,
        init_param=init_param,
        wd=wd,
        name=name,
        trainable=trainable,
        seed=seed)


@RegisterModel("resnet-gpu")
def get_resnet_model_gpu(*args, **kwargs):
  return ResNetModelGPU(*args, **kwargs)
