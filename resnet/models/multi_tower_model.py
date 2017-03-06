"""Multi-tower CNN for training parallel GPU jobs."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import tensorflow as tf

from resnet.models.resnet_model import ResNetModel
from resnet.utils import logger

log = logger.get()


class MultiTowerModel(object):

  def __init__(self,
               config,
               tower_cls,
               is_training=True,
               num_replica=2,
               inp=None,
               label=None,
               optimizer=None,
               apply_grad=True):
    self._config = config
    self._is_training = is_training
    self._num_replica = num_replica
    self._opt = optimizer
    self._apply_grad = apply_grad
    self._tower_cls = tower_cls

    # Input.
    if inp is None:
      x = tf.placeholder(
          self.dtype(), [None, config.height, config.width, config.num_channel])
    else:
      x = inp
    if label is None:
      y = tf.placeholder(tf.int32, [None])
    else:
      y = label

    self._input = x
    self._label = y
    self._towers = []
    self._build_towers()

  def _average_gradients(self, tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
      Note that this function provides a synchronization point across all towers.
      Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
      Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for g, v in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        if g is None:
          log.warning("No gradient for variable \"{}\"".format(v.name))
          grads.append(None)
          break
        else:
          expanded_g = tf.expand_dims(g, 0)
          grads.append(expanded_g)

      # Average over the "tower" dimension.
      if grads[0] is None:
        grad = None
      else:
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower"s pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads

  def dtype(self):
    tensor_type = os.getenv("TF_DTYPE", "float32")
    if tensor_type == "float32":
      return tf.float32
    elif tensor_type == "float64":
      return tf.float64
    else:
      raise Exception("Unknown tensor type {}".format(tensor_type))

  def assign_weights(self, weights):
    return self._towers[0].assign_weights(weights)

  def get_weights(self, compatible=False):
    return self._towers[0].get_weights(compatible=compatible)

  def _build_towers(self):
    # Calculate the gradients for each model tower.
    config = self.config
    tower_grads = []
    op_list = []

    with tf.device("/cpu:0"):
      inputs = tf.split(0, self.num_replica, self.input)
      labels = tf.split(0, self.num_replica, self.label)
      outputs = []
      costs = []
      cross_ents = []
      tower_grads = []

      if self.is_training:
        self._lr = tf.Variable(
            0.0, name="learn_rate", dtype=self.dtype(), trainable=False)
        if self._opt is None:
          opt = tf.train.MomentumOptimizer(
              self.lr, momentum=self.config.momentum)
        else:
          opt = self._opt

      for ii in xrange(self.num_replica):
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
        if visible_devices is None:
          device = "/cpu:0"
        else:
          num_gpu = len(visible_devices)
          device = "/gpu:{}".format(ii % num_gpu)
        #device = "/cpu:0"
        with tf.device(device):
          with tf.name_scope("%s_%d" % ("replica", ii)) as scope:
            tower_ = self._tower_cls(
                config,
                is_training=self.is_training,
                inference_only=True,
                inp=inputs[ii],
                label=labels[ii])
            outputs.append(tower_.output)
            cross_ents.append(tower_.cross_ent)
            costs.append(tower_.cost)
            self._towers.append(tower_)

            if self.is_training:
              # Calculate the gradients for the batch of data on this tower.
              wd_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
              if len(wd_losses) > 0:
                log.info("Replica {}, Weight decay variables: {}".format(
                    ii, wd_losses))
                log.info("Replica {}, Number of weight decay variables: {}".
                         format(ii, len(wd_losses)))
              grads = opt.compute_gradients(tower_.cost)
              tower_grads.append(grads)

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

      self._output = tf.concat(0, outputs)
      self._cost = tf.reduce_mean(tf.pack(costs))
      self._cross_ent = tf.reduce_mean(tf.pack(cross_ents))
      grads = self._average_gradients(tower_grads)
      self._grads = grads

      if not self.is_training:
        return

      global_step = tf.Variable(0.0, trainable=False)
      ### Should have named it like this.
      # global_step = tf.Variable(0.0, name="global_step", trainable=False)
      self._global_step = global_step

      if self._apply_grad:
        train_op = opt.apply_gradients(grads, global_step=global_step)
        self._train_op = train_op
      self._new_lr = tf.placeholder(
          self.dtype(), shape=[], name="new_learning_rate")
      self._lr_update = tf.assign(self._lr, self._new_lr)

  @property
  def input(self):
    return self._input

  @property
  def output(self):
    return self._output

  @property
  def label(self):
    return self._label

  @property
  def grads(self):
    return self._grads

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
  def cost(self):
    return self._cost

  @property
  def cross_ent(self):
    return self._cross_ent

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

  @property
  def global_step(self):
    return self._global_step

  def assign_lr(self, session, lr_value):
    """Assigns new learning rate."""
    log.info("Adjusting learning rate to {}".format(lr_value))
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  def infer_step(self, sess, inp):
    """Run inference."""
    return sess.run(self.output, feed_dict={self.input: inp})

  def train_step(self, sess, inp, label):
    """Run training."""
    feed_data = {self.input: inp, self.label: label}
    cost, ce, _ = sess.run([self.cost, self.cross_ent, self.train_op],
                           feed_dict=feed_data)
    return ce
