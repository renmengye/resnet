"""Multi-pass multi-tower CNN for training parallel GPU jobs."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
import tensorflow as tf

from resnet.models.multi_pass_optimizer import MultiPassOptimizer
from resnet.utils import logger

log = logger.get()


class MultiPassModel(object):
  """This model can average gradients from serial forward-backward propagations.
  """

  def __init__(self,
               config,
               model,
               is_training=True,
               num_passes=2,
               debug=False,
               aggregate_method="cumsum"):
    self._config = config
    self._debug = debug
    self._aggregate_method = aggregate_method
    self._model = model
    self._is_training = is_training
    self._num_passes = num_passes
    self._train_op_list = []
    self._build_optimizer()

  @property
  def dtype(self):
    tensor_type = os.getenv("TF_DTYPE", "float32")
    if tensor_type == "float32":
      return tf.float32
    else:
      return tf.float64

  @property
  def model(self):
    return self._model

  @property
  def train_op_list(self):
    return self._train_op_list

  def assign_weights(self, weights):
    return self.model.assign_weights(weights)

  def get_weights(self):
    return self.model.get_weights()

  def _build_optimizer(self):
    config = self.config
    if not self.is_training:
      return

    self._lr = tf.get_variable(
        "learn_rate", [],
        initializer=tf.constant_initializer(0.0),
        dtype=self.dtype,
        trainable=False)
    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)
    opt = tf.train.MomentumOptimizer(self.lr, momentum=config.momentum)
    opt = MultiPassOptimizer(
        opt,
        num_passes=self.num_passes,
        debug=self._debug,
        aggregate_method=self._aggregate_method)
    self._optimizer = opt

    tf.get_variable_scope()._reuse = None
    global_step = tf.get_variable(
        "global_step", [],
        initializer=tf.constant_initializer(0.0),
        trainable=False,
        dtype=self.dtype)
    self._global_step = global_step

    # Add all trainable variables to the variable list.
    for ii in range(self.num_passes):
      self._train_op_list.append(
          opt.apply_gradients(
              self.model.grads_and_vars, global_step=global_step))

  def _slice_data(self, data, idx):
    num_per_pass = int(np.ceil(data.shape[0] / self.num_passes))
    start = idx * num_per_pass
    end = min(start + num_per_pass, data.shape[0])
    return data[start:end]

  def assign_lr(self, session, lr_value):
    """Assigns new learning rate."""
    log.info("Adjusting learning rate to {}".format(lr_value))
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def train_step(self, sess, inp=None, label=None):
    """Run training."""
    ce = 0.0
    # TODO: Currently we do not support in graph inputs. To do this, we will
    # have to build the subgraph internally within this class (see
    # MultiTowerModel).
    assert inp is not None, "Non-placeholder not supported yet."
    assert label is not None, "Non-placeholder not supported yet."
    for ii, train_op in enumerate(self.train_op_list):
      feed_data = {
          self.input: self._slice_data(inp, ii),
          self.label: self._slice_data(label, ii)
      }
      results = sess.run(
          [self.model.cross_ent, train_op] + self.model.bn_update_ops,
          feed_dict=feed_data)
      ce += results[0] / self.num_passes
    return ce

  def infer_step(self, sess, inp=None):
    """Run inference."""
    assert inp is not None, "Non-placeholder not supported yet."
    _feed_data = {self.model.input: inp}
    return sess.run(self.model.output, feed_dict=_feed_data)

  @property
  def input(self):
    return self._model.input

  @property
  def output(self):
    return self._model.output

  @property
  def label(self):
    return self._model.label

  @property
  def cost(self):
    return self._model.cost

  @property
  def cross_ent(self):
    return self._model.cross_ent

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def num_passes(self):
    return self._num_passes

  @property
  def global_step(self):
    return self._global_step

  @property
  def config(self):
    return self._config

  @property
  def is_training(self):
    return self._is_training

  @property
  def num_replica(self):
    return self._num_replica

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op
