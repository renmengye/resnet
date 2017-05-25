"""Multi-tower CNN for training parallel GPU jobs, using NCCL.
NCCL will be useful on cluster such as DGX-1, with high bandwith
interconnection.
This class is not recommended to use for clusters with 4 Titan-X's with
standard PCI connections.
In this model, each GPU will maintain an individual copy of the variables, and
the gradients are synced through NCCL all_sum function, and individually
applied to each copy.
For serialization, additional wrapper is needed to convert the replica
sensitive namings to replica agnostic naming (strip "replica_%d" in the names
of the variables).

WARNING: This class currently does not compute the correct exponential moving
averages for BN. To do this correctly, one needs to create a variable on CPU
and compute the average of all of the towers. It will be a question whether the
average happens during training or purely during inference.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import tensorflow as tf

from resnet.models.nnlib import concat, split, stack
from resnet.utils import logger
from resnet.models.multi_tower_model import MultiTowerModel
from tensorflow.contrib.nccl import all_sum

log = logger.get()


class MultiTowerModelNCCL(MultiTowerModel):

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
    num_tower = len(tower_grads)
    average_grads = []
    for ii in range(num_tower):
      average_grads.append([])

    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = [gv[0] for gv in grad_and_vars]

      # Average over the "tower" dimension, using NCCL.
      if grads[0] is not None:
        #print(grads)
        #[print(g.device) for g in grads]
        grads = all_sum(grads)

      # Store averaged gradients for every tower.
      for ii in range(num_tower):
        with tf.device(self._get_replica_device(ii)):
          average_grads[ii].append(
              (grads[ii] / float(num_tower), grad_and_vars[ii][1]))
    return average_grads

  def _get_device(self, device_name="cpu", device_id=0):
    return "/{}:{:d}".format(device_name, device_id)

  def _get_replica_device(self, replica_id):
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if visible_devices is None:
      device = self._get_device("cpu", 0)
    else:
      num_gpu = len(visible_devices)
      device = self._get_device("gpu", replica_id % num_gpu)
    return device

  def _build_towers(self):
    # Calculate the gradients for each model tower.
    config = self.config
    tower_grads = []
    op_list = []

    with tf.device(self._get_device("cpu", 0)):
      inputs = split(self.input, self.num_replica, axis=0)
      labels = split(self.label, self.num_replica, axis=0)
      outputs = []
      costs = []
      cross_ents = []
      tower_grads_and_vars = []

      if self.is_training:
        self._lr = tf.get_variable(
            "learn_rate", [],
            initializer=tf.constant_initializer(0.0),
            dtype=self.dtype,
            trainable=False)

      all_vars = set(tf.trainable_variables())
      var_names_sorted = None
      for ii in range(self.num_replica):
        with tf.device(self._get_replica_device(ii)):
          # Here name_scope is changed to variable_scope.
          with tf.variable_scope("%s_%d" % ("replica", ii)) as scope:
            tower_ = self._tower_cls(
                config,
                is_training=self.is_training,
                inference_only=True,
                inp=inputs[ii],
                label=labels[ii],
                batch_size=self._batch_size,
                idx=ii)
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
              
              cur_vars = tf.trainable_variables()
              rep_vars = filter(lambda x: x not in all_vars, cur_vars)
              rep_vars_dict = dict(
                  map(lambda x: (x.name.replace("replica_{:d}".format(ii), "replica_0"), x),
                      rep_vars))
              if var_names_sorted is None:
                var_names_sorted = sorted(map(lambda x: x.name, rep_vars))
              rep_vars = map(lambda x: rep_vars_dict[x], var_names_sorted)
              all_vars = set(cur_vars)
              tower_grads_and_vars.append(
                  tower_._compute_gradients(
                      tower_.cost, var_list=rep_vars))

            log.info("Replica {} built".format(ii), verbose=0)
            ## Reuse variables for the next tower.
            # No reuse varaibles now. All variables are replicated.
            #tf.get_variable_scope().reuse_variables()

      self._output = concat(outputs, axis=0)
      self._output_idx = tf.cast(tf.argmax(self._output, axis=1), tf.int32)
      self._correct = tf.to_float(tf.equal(self._output_idx, self.label))
      self._cost = tf.reduce_mean(stack(costs))
      self._cross_ent = tf.reduce_mean(stack(cross_ents))
      if not self.is_training:
        return

      grads_and_vars = self._average_gradients(tower_grads_and_vars)
      self._tower_grads_and_vars = tower_grads_and_vars
      self._grads_and_vars = grads_and_vars[0]
      train_op_list = []
      self._new_lr = tf.placeholder(
          self.dtype, shape=[], name="new_learning_rate")
      self._lr_update = tf.assign(self._lr, self._new_lr)

      if self._apply_grad:
        #tf.get_variable_scope()._reuse = None
        global_step = tf.get_variable(
            "global_step", [],
            initializer=tf.constant_initializer(0.0),
            trainable=False,
            dtype=self.dtype)
        self._global_step = global_step
        for ii in range(self.num_replica):
          with tf.device(self._get_replica_device(ii)):
            opt = tf.train.MomentumOptimizer(
                self.lr, momentum=self.config.momentum)
            train_op = opt.apply_gradients(grads_and_vars[ii])
            train_op_list.append(train_op)
        with tf.control_dependencies(train_op_list):
          self._train_op = tf.assign_add(global_step, 1.0)
