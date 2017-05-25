"""Multi-pass multi-tower CNN for training parallel GPU jobs.
Difference from V1 is that, this version can take external inp and label nodes
instead of relying on splitting the data in NumPy during run time.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
import tensorflow as tf

from resnet.models.multi_pass_model import MultiPassModel
from resnet.models.nnlib import split


class MultiPassModelV2(MultiPassModel):
  """This model can average gradients from serial forward-backward propagations.
  """

  def __init__(self,
               config,
               model_cls,
               is_training=True,
               num_passes=2,
               inp=None,
               label=None,
               batch_size=None,
               aggregate_method="cumsum",
               debug=False):
    self._config = config
    self._model_cls = model_cls
    self._aggregate_method = aggregate_method
    self._debug = debug
    # Input.
    if inp is None:
      x = tf.placeholder(
          self.dtype,
          [batch_size, config.height, config.width, config.num_channel],
          name="x")
    else:
      x = inp
    if label is None:
      y = tf.placeholder(tf.int32, [batch_size], name="y")
    else:
      y = label

    self._pass_id = tf.placeholder(tf.int32, [], name="pass_id")
    self._input = x
    #self._input_list = split(x, num_passes, 0)
    # Make sure that the labels are in reasonable range.
    # with tf.control_dependencies(
    #     [tf.assert_greater_equal(y, 0), tf.assert_less(y, config.num_classes)]):
    #   self._label = tf.identity(y)
    self._label = y
    #self._label_list = split(y, num_passes, 0)
    self._model = None
    self._is_training = is_training
    self._num_passes = num_passes
    self._train_op_list = []
    self._batch_size = batch_size
    self._build_inference()
    self._build_optimizer()

  def _build_inference(self):
    inp, label = self._slice_inp(self.input, self.label, self._pass_id)
    self._model = self._model_cls(
        self.config,
        is_training=self.is_training,
        inference_only=False,
        inp=inp,
        label=label,
        batch_size=self._batch_size,
        apply_grad=False)

  def _slice_inp(self, inp, label, idx):
    batch_size = tf.cast(tf.shape(inp)[0], tf.int32)
    num_per_pass = tf.cast(batch_size / self._num_passes, tf.int32)
    start = idx * num_per_pass
    return tf.slice(inp, [start, 0, 0, 0],
                    [num_per_pass, -1, -1, -1]), tf.slice(label, [start],
                                                          [num_per_pass])

  def train_step(self, sess, inp=None, label=None):
    """Run training."""
    ce = 0.0
    for ii, train_op in enumerate(self.train_op_list):
      if inp is not None:
        feed_data = {
            self.input: self._slice_data(inp, ii),
            self.label: self._slice_data(label, ii)
        }
      else:
        feed_data = dict()
      feed_data[self._pass_id] = ii
      results = sess.run(
          [self.model.cross_ent, train_op, self.model.bn_update_ops],
          feed_dict=feed_data)
      ce += results[0] / self.num_passes
    return ce

  def infer_step(self, sess, inp=None):
    """Run inference."""
    if inp is None:
      _feed_data = {self.model.input: inp}
    return sess.run(self.model.output, feed_dict=_feed_data)

  @property
  def input(self):
    return self._input

  @property
  def label(self):
    return self._label
