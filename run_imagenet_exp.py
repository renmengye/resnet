#!/usr/bin/env python
"""
Authors: Mengye Ren (mren@cs.toronto.edu) Renjie Liao (rjliao@cs.toronto.edu)

The following code explores different normalization schemes in CNN on ImageNet
datasets.

Usage:
python run_imagenet_exp.py --model           [MODEL NAME]        \
    --config          [CONFIG FILE]       \
    --data_folder     [DATASET FOLDER]    \
    --logs            [LOGS FOLDER]       \
    --results         [SAVE FOLDER]       \
    --num_gpu         [NUMBER OF GPU]

Flags:
    --model: Model type. Available options are:
         1) resnet-50
         2) resnet-101
         3) resnet-152
         4) resnet-50-dn
    --config: Not using the pre-defined configs above, specify the JSON file
    that contains model configurations.
    --dataset: Dataset name. Available options are: 1) cifar-10 2) cifar-100.
    --data_folder: Path to data folder, default is ../data/{DATASET}.
    --logs: Path to logs folder, default is ../logs/default.
    --results: Path to save folder, default is ../results.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import os
import tensorflow as tf

from resnet.data import ImageNetDataset
from resnet.utils import BatchIterator, ConcurrentBatchIterator
from resnet.utils import logger
from resnet.utils import progress_bar as pb

import imagenet_exp_config as conf
from multi_tower_cnn_model import MultiTowerCNNModel
from multi_pass_multi_tower_cnn_model import MultiPassMultiTowerCNNModel
from cifar_exp_logger import ExperimentLogger

log = logger.get()

flags = tf.flags
flags.DEFINE_string("config", None, "manually defined config file")
flags.DEFINE_string("id", None, "experiment ID")
flags.DEFINE_string("results", "../results/imagenet", "saving folder")
flags.DEFINE_string("logs", "../logs/default", "logging folder")
flags.DEFINE_string("model", "vgg-16", "model type")
flags.DEFINE_bool("restore", False, "Restore checkpoint")
flags.DEFINE_integer("max_num_steps", -1, "Maximum number of steps")
FLAGS = flags.FLAGS

DATASET = "imagenet"
DATASET_FOLDER = "../data/imagenet"


def get_config():
  # Manually set config.
  if FLAGS.config is not None:
    return conf.BaselineConfig.from_json(open(FLAGS.config, "r").read())
  else:
    return conf.get_config(DATASET, FLAGS.model)


def get_dataset(folder, split, mode):
  """Gets ImageNet dataset.

  Args:
    folder: Dataset folder.
    split: "train", "valid", or "test".
    mode: "train", "valid".

  Returns:
    dp: Dataset object.
  """
  return ImageNetDataset(folder, split, mode=mode)


def get_iter(dataset,
             batch_size=256,
             shuffle=False,
             cycle=False,
             log_epoch=-1,
             seed=0,
             prefetch=True,
             num_worker=20,
             queue_size=50,
             num_batches=-1):
  """Gets a data iterator.

  Args:
    dataset: Dataset object.
    batch_size: Mini-batch size.
    shuffle: Whether to shuffle the data.
    cycle: Whether to stop after one full epoch.
    log_epoch: Log progress after how many iterations.

  Returns:
    b: Batch iterator object.
  """
  b = BatchIterator(
      dataset.get_size(),
      batch_size=batch_size,
      shuffle=shuffle,
      cycle=cycle,
      get_fn=dataset.get_batch_idx,
      log_epoch=log_epoch,
      seed=seed,
      num_batches=num_batches)
  if prefetch:
    b = ConcurrentBatchIterator(
        b,
        max_queue_size=queue_size,
        num_threads=num_worker,
        log_queue=-1,
        name=dataset.split)
    return b


class FixedLearnRateScheduler(object):

  def __init__(self, sess, model, base_lr, lr_decay_steps, lr_list=None):
    self.model = model
    self.sess = sess
    self.lr = base_lr
    self.lr_list = lr_list
    self.lr_decay_steps = lr_decay_steps
    self.model.assign_lr(self.sess, self.lr)

  def step(self, niter):
    if len(self.lr_decay_steps) > 0:
      if (niter + 1) == self.lr_decay_steps[0]:
        if self.lr_list is not None:
          self.lr = self.lr_list[0]
        else:
          self.lr *= 0.1  ## Divide 10 by default!!!
        self.model.assign_lr(self.sess, self.lr)
        self.lr_decay_steps.pop(0)
        log.warning("LR decay steps {}".format(self.lr_decay_steps))
        if self.lr_list is not None:
          self.lr_list.pop(0)
      elif (niter + 1) > self.lr_decay_steps[0]:
        ls = self.lr_decay_steps
        while len(ls) > 0 and (niter + 1) > ls[0]:
          ls.pop(0)
          log.warning("LR decay steps {}".format(self.lr_decay_steps))
          if self.lr_list is not None:
            self.lr = self.lr_list.pop(0)
          else:
            self.lr *= 0.1
        self.model.assign_lr(self.sess, self.lr)


class DynamicLearnRateScheduler(object):

  def __init__(self, sess, model, base_lr, max_num_decays):
    self.model = model
    self.sess = sess
    self.lr = base_lr
    self.max_num_decays = max_num_decays
    self.num_decays = 0
    self.model.assign_lr(self.sess, self.lr)
    self.prev_valid_acc = 0.0

  def step(self, valid_acc):
    if self.num_decays < self.max_num_decays:
      if valid_acc < self.prev_valid_acc:
        self.lr *= 0.1
        self.model.assign_lr(self.sess, self.lr)
        self.num_decays += 1
    self.prev_valid_acc = valid_acc


def get_model(config, num_replica, is_training):
  if hasattr(config, "num_passes"):
    num_passes = config.num_passes
  else:
    num_passes = 1

  if num_passes > 1:
    return MultiPassMultiTowerCNNModel(
        config,
        num_replica=num_replica,
        is_training=is_training,
        num_passes=num_passes,
        compatible=True)
  else:
    return MultiTowerCNNModel(
        config, num_replica=num_replica, is_training=is_training)


def train_model(config, environ, train_data, trainval_data=None,
                test_data=None):
  """Trains a CIFAR model.

  Args:
    config: Config object
    environ: Environ object
    train_data: Dataset object
    test_data: Dataset object

  Returns:
    acc: Final test accuracy
  """
  np.random.seed(0)
  tf.set_random_seed(1234)
  if environ.verbose:
    verbose_level = 0
  else:
    verbose_level = 2

  log.info("Environment: {}".format(environ.__dict__))
  log.info("Config: {}".format(config.__dict__))

  save_folder = os.path.join(environ.save_folder, environ.exp_id)
  logs_folder = os.path.join(environ.logs_folder, environ.exp_id)
  with log.verbose_level(verbose_level):
    exp_logger = ExperimentLogger(logs_folder)

    # Gets data iterators.
    train_iter = get_iter(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        cycle=True,
        prefetch=config.prefetch,
        num_worker=25,
        queue_size=100)

    if trainval_data is not None:
      trainval_iter = get_iter(
          trainval_data,
          batch_size=config.valid_batch_size,
          shuffle=True,
          cycle=True,
          prefetch=config.prefetch,
          num_worker=5,
          queue_size=10)

    if test_data is not None:
      test_iter = get_iter(
          test_data,
          batch_size=config.valid_batch_size,
          shuffle=True,
          cycle=False,
          prefetch=config.prefetch,
          num_worker=25,
          queue_size=100)

    # Builds models.
    log.info("Building models")
    with tf.name_scope("Train"):
      with tf.variable_scope("Model", reuse=None):
        m = get_model(config, num_replica=environ.num_gpu, is_training=True)

    if trainval_data is not None or test_data is not None:
      with tf.name_scope("Valid"):
        with tf.variable_scope("Model", reuse=True):
          mvalid = get_model(
              config, num_replica=environ.num_gpu, is_training=False)

    # Initializes variables.
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver = tf.train.Saver(max_to_keep=None)  ### Keep all checkpoints here!
      if FLAGS.restore:
        log.info("Restore checkpoint \"{}\"".format(save_folder))
        saver.restore(sess, tf.train.latest_checkpoint(save_folder))
      else:
        sess.run(tf.initialize_all_variables())

      def train_step():
        """Train step."""
        batch = train_iter.next()
        return m.train_step(sess, batch["img"], batch["label"])

      def save():
        """Snapshots a model."""
        if not os.path.isdir(save_folder):
          os.makedirs(save_folder)
          config_file = os.path.join(save_folder, "conf.json")
          environ_file = os.path.join(save_folder, "env.json")
          with open(config_file, "w") as f:
            f.write(config.to_json())
          with open(environ_file, "w") as f:
            f.write(environ.to_json())
        log.info("Saving to {}".format(save_folder))
        saver.save(
            sess,
            os.path.join(save_folder, "model.ckpt"),
            global_step=m.global_step)

      def evaluate(data_iter, nbatches):
        """Runs evaluation."""
        num_correct = 0.0
        count = 0
        if nbatches == -1:
          iter_ = data_iter
        else:
          iter_ = range(nbatches)

        for bb in iter_:
          if nbatches == -1:
            batch = bb
          else:
            batch = data_iter.next()
          y = mvalid.infer_step(sess, batch["img"])
          pred_label = np.argmax(y, axis=1)
          num_correct += np.sum(
              np.equal(pred_label, batch["label"]).astype(float))
          count += pred_label.size
        acc = (num_correct / count)
        return acc

      def train():
        """Train loop."""
        # Learn rate scheduler.
        lr = config.base_learn_rate
        if config.learn_rate_schedule == "fixed":
          if hasattr(config, "lr_list"):
            lr_list = config.lr_list
          else:
            lr_list = None  # By default will divide the learning rate by 10.
          lr_scheduler = FixedLearnRateScheduler(
              sess, m, lr, config.lr_decay_steps, lr_list=lr_list)
        elif config.learn_rate_schedule == "dynamic":
          # Don't use this for now...
          lr_scheduler = DynamicLearnRateScheduler(sess, m, lr,
                                                   config.max_num_lr_decays)
        max_train_iter = config.max_train_iter
        niter_start = int(sess.run(m.global_step))

        # Add upper bound to the number of steps.
        if FLAGS.max_num_steps > 0:
          max_train_iter = min(max_train_iter,
                               niter_start + FLAGS.max_num_steps)

        if environ.verbose:
          loop = range(niter_start, max_train_iter)
        else:
          loop = pb.get_iter(range(niter_start, max_train_iter))

        for niter in loop:
          # Decrease learn rate.
          if config.learn_rate_schedule == "fixed":
            lr_scheduler.step(niter)

          # Train step.
          ce = train_step()

          # Log training loss.
          if (niter + 1) % config.disp_iter == 0 or niter == 0:
            exp_logger.log_train_ce(niter, ce)

          if (niter + 1) % (config.disp_iter * 10) == 0 or niter == 0:
            log.info("Experment ID {}".format(environ.exp_id))

          # Log training acc.
          if trainval_data is not None:
            if (niter + 1) % config.trainval_iter == 0 or niter == 0:
              acc = evaluate(trainval_iter, config.num_trainval_batch)
              exp_logger.log_train_acc(niter, acc)

          # Log validation acc.
          if test_data is not None:
            if (niter + 1) % config.valid_iter == 0 or niter == 0:
              log.info("Running validation")
              test_iter.reset()
              acc = evaluate(test_iter, config.num_valid_batch)
              exp_logger.log_valid_acc(niter, acc)
              # Decrease learn rate if validation accuracy goes down.
              if config.learn_rate_schedule == "dynamic":
                lr_scheduler.step(acc)
          # Save model.
          # if (niter + 1) % config.save_iter == 0 or niter == 0:
          if (niter + 1) % config.save_iter == 0:
            save()

        if test_data is not None:
          test_iter.reset()
          acc = evaluate(test_iter, -1)
          log.info("Final test accuracy = {:.3f}".format(acc * 100))
          return acc
        else:
          return None

      acc = train()
  if test_data is not None:
    return acc
  else:
    return None


def main():
  # Loads parammeters.
  config = get_config()
  environ = get_environ()

  # Configures dataset objects.
  log.info("Building dataset")
  train_data = get_dataset(environ.data_folder, "train", "train")
  if config.run_validation:
    trainval_data = get_dataset(environ.data_folder, "train", "valid")
    test_data = get_dataset(environ.data_folder, "valid", "valid")
  else:
    trainval_data = None
    test_data = None

  # Trains a model.
  train_model(config, environ, train_data, trainval_data, test_data)


if __name__ == "__main__":
  main()
