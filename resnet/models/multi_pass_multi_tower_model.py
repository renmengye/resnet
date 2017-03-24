"""Multi-pass multi-tower CNN for training parallel GPU jobs."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from resnet.models.resnet_model import ResNetModel
from resnet.models.multi_tower_model import MultiTowerModel
from resnet.models.multi_pass_optimizer import MultiPassOptimizer
from resnet.utils import logger

log = logger.get()


class MultiPassMultiTowerModel(object):

  def __init__(self,
               config,
               tower_cls,
               num_replica=2,
               is_training=True,
               num_passes=2,
               inp=None,
               label=None,
               optimizer=None):
    self._config = config
    self._tower_cls = tower_cls
    self._is_training = is_training
    self._num_replica = num_replica
    self._num_passes = num_passes

    # Input.
    if inp is None:
      x = tf.placeholder(
          tf.float32, [None, config.height, config.width, config.num_channel])
    else:
      x = inp
    if label is None:
      y = tf.placeholder(tf.int32, [None])
    else:
      y = label

    self._input = x
    self._label = y
    self._model = None
    self._train_op_list = []
    self._build_towers()

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

  def _build_towers(self):
    config = self.config
    _towers = []
    if self.is_training:
      self._lr = tf.Variable(0.0, name="learn_rate", trainable=False)
      self._new_lr = tf.placeholder(
          tf.float32, shape=[], name="new_learning_rate")
      self._lr_update = tf.assign(self._lr, self._new_lr)
      opt = tf.train.MomentumOptimizer(self.lr, momentum=config.momentum)
      opt = MultiPassOptimizer(opt, num_passes=self.num_passes)
    else:
      opt = None
    self._model = MultiTowerModel(
        config,
        self._tower_cls,
        is_training=self._is_training,
        num_replica=self.num_replica,
        optimizer=opt,
        apply_grad=False)
    if not self.is_training:
      return
    global_step = tf.Variable(0.0, name="global_step", trainable=False)
    self._global_step = global_step
    for ii in range(self.num_passes):
      self._train_op_list.append(
          opt.apply_gradients(
              self.model.grads, global_step=global_step))

  def _slice_data(self, data, idx):
    num_per_pass = int(np.ceil(data.shape[0] / self.num_passes))
    start = idx * num_per_pass
    end = min(start + num_per_pass, data.shape[0])
    return data[start:end]

  def assign_lr(self, session, lr_value):
    """Assigns new learning rate."""
    log.info("Adjusting learning rate to {}".format(lr_value))
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def train_step(self, sess, inp, label):
    """Run training."""
    ce = 0.0
    for ii, train_op in enumerate(self.train_op_list):
      #print(self._slice_data(inp, ii).shape)
      _feed_data = {
          self.model.input: self._slice_data(inp, ii),
          self.model.label: self._slice_data(label, ii)
      }
      _cost, _ce, _ = sess.run(
          [self.model.cost, self.model.cross_ent, train_op],
          feed_dict=_feed_data)
      ce += _ce / self.num_passes
    return ce

  def infer_step(self, sess, inp):
    """Run inference."""
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
  def opt(self):
    return self._opt

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


def get_diff_signature(w, dw):
  return dw - w


def print_w(name, w, num_elem=3):
  log.info((name, w.ravel()[:num_elem]))


# TODO: make this a unit test.
def test_multi_pass():
  import cifar_exp_config as cifar_conf
  from data import CIFAR10Dataset
  from utils import BatchIterator
  import os
  if os.path.exists("/ais/gobi4"):
    folder = "/ais/gobi4/mren/data/cifar-10"
  else:
    folder = "/home/mren/data/cifar-10"
  data = CIFAR10Dataset(folder=folder, split="valid")
  config = cifar_conf.BaselineConfig()
  b = BatchIterator(
      data.get_size(),
      batch_size=8,
      shuffle=False,
      cycle=False,
      get_fn=data.get_batch_idx)

  # Testing the batch iterator.
  b1 = b.next()
  b.reset()
  b2 = b.next()
  np.testing.assert_almost_equal(b1["img"], b2["img"])
  b.reset()
  config.pool_fn = ["avg_pool", "avg_pool", "avg_pool"]

  num_rep = 4
  num_pas = 2
  learn_rate = 1.0
  decimal_tol = 5
  num_elem_dbg = 3
  wlist = [
      "mlp/layer_1/w", "mlp/layer_0/w", "cnn/layer_2/w", "cnn/layer_1/w",
      "cnn/layer_0/w", "mlp/layer_1/b", "mlp/layer_0/b", "cnn/layer_2/b",
      "cnn/layer_1/b", "cnn/layer_0/b"
  ]

  for wname in wlist:
    with log.verbose_level(2):
      ######################################
      # Run the MultiPass model.
      ######################################
      with tf.Graph().as_default():
        s1 = tf.Session()
        with tf.variable_scope("Model"):
          m1 = MultiPassMultiTowerModel(
              config, num_replica=num_rep, num_passes=num_pas)
        tf.set_random_seed(1234)
        s1.run(tf.initialize_all_variables())
        m1.assign_lr(s1, learn_rate)
        batch = b.next()
        with tf.variable_scope("Model", reuse=True):
          w1 = s1.run(tf.get_variable(wname))
        ce1 = m1.train_step(s1, batch["img"], batch["label"])
        with tf.variable_scope("Model", reuse=True):
          w1d = s1.run(tf.get_variable(wname))
      b.reset()

      ######################################
      # Run the regular MultiTower model.
      ######################################
      with tf.Graph().as_default():
        s2 = tf.Session()
        with tf.variable_scope("Model2") as scope:
          m2 = MultiTowerModel(config, num_replica=num_rep)
        tf.set_random_seed(1234)
        s2.run(tf.initialize_all_variables())
        m2.assign_lr(s2, learn_rate)
        with tf.variable_scope("Model2", reuse=True):
          w2 = s2.run(tf.get_variable(wname))
        ce2 = m2.train_step(s2, batch["img"], batch["label"])
        with tf.variable_scope("Model2", reuse=True):
          w2d = s2.run(tf.get_variable(wname))
      b.reset()

      ######################################
      # Run the regular model.
      ######################################
      with tf.Graph().as_default():
        s3 = tf.Session()
        with tf.variable_scope("Model3") as scope:
          m3 = CNNModel(config)
        tf.set_random_seed(1234)
        s3.run(tf.initialize_all_variables())
        m3.assign_lr(s3, learn_rate)
        with tf.variable_scope("Model3", reuse=True):
          w3 = s3.run(tf.get_variable(wname))
        ce3 = m3.train_step(s3, batch["img"], batch["label"])
        with tf.variable_scope("Model3", reuse=True):
          w3d = s3.run(tf.get_variable(wname))
      b.reset()

    # Make this block one indent level to avoid logging.
    ######################################
    # Make sure the weights are the same.
    ######################################
    log.info("Testing {}".format(wname))
    print_w("w1", w1, num_elem_dbg)
    print_w("w2", w2, num_elem_dbg)
    print_w("w3", w3, num_elem_dbg)
    np.testing.assert_almost_equal(w1, w2, decimal=decimal_tol)
    np.testing.assert_almost_equal(w2, w3, decimal=decimal_tol)

    ######################################
    # Make sure the gradients are the same. 
    ######################################
    print_w("w1 delta", w1d - w1, num_elem_dbg)
    print_w("w2 delta", w2d - w2, num_elem_dbg)
    print_w("w3 delta", w3d - w3, num_elem_dbg)
    print_w("w1 new", w1d, num_elem_dbg)
    print_w("w2 new", w2d, num_elem_dbg)
    print_w("w3 new", w3d, num_elem_dbg)

    np.testing.assert_almost_equal(
        get_diff_signature(w1, w1d),
        get_diff_signature(w2, w2d),
        decimal=decimal_tol)
    np.testing.assert_almost_equal(
        get_diff_signature(w2, w2d),
        get_diff_signature(w3, w3d),
        decimal=decimal_tol)
    log.info("Success")