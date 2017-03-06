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

import imagenet_exp_config as conf
from resnet.models import MultiTowerModel
from resnet.models import MultiPassMultiTowerModel
from cifar_exp_logger import ExperimentLogger

log = logger.get()

flags = tf.flags
flags.DEFINE_string("config", None, "manually defined config file")
flags.DEFINE_string("id", None, "experiment ID")
flags.DEFINE_string("./results", "results/imagenet", "saving folder")
flags.DEFINE_string("./logs", "logs/default", "logging folder")
flags.DEFINE_string("model", "resnet-50", "model type")
flags.DEFINE_bool("restore", False, "Restore checkpoint")
flags.DEFINE_integer("max_num_steps", -1, "Maximum number of steps")
flags.DEFINE_integer("num_gpu", 4, "Number of GPUs")
flags.DEFINE_integer("num_pass", 1, "Number of forward-backwad passes")
FLAGS = flags.FLAGS

DATASET = "imagenet"


def get_config():
  # Manually set config.
  if FLAGS.config is not None:
    return conf.BaselineConfig.from_json(open(FLAGS.config, "r").read())
  else:
    return conf.get_config(DATASET, FLAGS.model)


def get_model(config, num_replica, num_pass, is_training):
  if num_replica > 1:
    if num_pass > 1:
      return MultiPassMultiTowerModel(
          config,
          ResNetModel,
          num_replica=num_replica,
          is_training=is_training,
          num_passes=num_pass,
          compatible=True)
    else:
      return MultiTowerModel(
          config, ResNetModel, num_replica=num_replica, is_training=is_training)
  elif num_replica == 1:
    return ResNetModel(config, is_training=is_training)
  else:
    raise Exception("Unacceptable number of replica: {}".format(num_replica))


def train_step(sess, model, batch):
  """Train step."""
  feed_data = {model.input: batch["img"], model.label: batch["label"]}
  cost, ce, _ = sess.run([model.cost, model.cross_ent, model.train_op],
                         feed_dict=feed_data)
  return ce


def train_model(exp_id, config, train_iter):
  """Trains a CIFAR model.

  Args:
    exp_id: String. Experiment ID.
    config: Config object
    train_data: Dataset object

  Returns:
    acc: Final test accuracy
  """
  log.info("Config: {}".format(config.__dict__))
  exp_logger = ExperimentLogger(logs_folder)

  # Initializes variables.
  with tf.Graph().as_default():
    np.random.seed(0)
    tf.set_random_seed(1234)

    # Builds models.
    log.info("Building models")
    with tf.name_scope("Train"):
      with tf.variable_scope("Model", reuse=None):
        m = get_model(
            config,
            num_replica=FLAGS.num_gpu,
            num_pass=FLAGS.num_pass,
            is_training=True)

    with tf.Session() as sess:
      saver = tf.train.Saver(max_to_keep=None)  ### Keep all checkpoints here!
      if FLAGS.restore:
        log.info("Restore checkpoint \"{}\"".format(save_folder))
        saver.restore(sess, tf.train.latest_checkpoint(save_folder))
      else:
        sess.run(tf.global_variables_initializer())

      max_train_iter = config.max_train_iter
      niter_start = int(m.global_step.eval())

      # Add upper bound to the number of steps.
      if FLAGS.max_num_steps > 0:
        max_train_iter = min(max_train_iter, niter_start + FLAGS.max_num_steps)

      # Set up learning rate schedule.
      lr = config.base_learn_rate
      lr_scheduler = FixedLearnRateScheduler(
          sess, m, lr, config.lr_decay_steps, lr_list=config.lr_list)

      for niter in tqdm(range(niter_start, config.max_train_iter), desc=exp_id):
        lr_scheduler.step(niter)
        ce = train_step(sess, m, train_iter.next())

        if (niter + 1) % config.disp_iter == 0 or niter == 0:
          exp_logger.log_train_ce(niter, ce)

        if (niter + 1) % config.save_iter == 0:
          if save_folder is not None:
            save(sess, saver, m.global_step, config, save_folder)


def main():
  # Loads parammeters.
  config = get_config()

  if FLAGS.id is None:
    exp_id = "exp_" + DATASET + "_" + FLAGS.model
    exp_id = gen_id(exp_id)

  if FLAGS.results is not None:
    save_folder = os.path.realpath(
        os.path.abspath(os.path.join(FLAGS.results, exp_id)))
    if not os.path.exists(save_folder):
      os.makedirs(save_folder)
  else:
    save_folder = None

  if FLAGS.logs is not None:
    logs_folder = os.path.realpath(
        os.path.abspath(os.path.join(FLAGS.logs, exp_id)))
    if not os.path.exists(logs_folder):
      os.makedirs(logs_folder)
  else:
    logs_folder = None

  # Configures dataset objects.
  log.info("Building dataset")
  train_data = get_dataset(DATASET, "train")

  # Trains a model.
  train_model(
      exp_id,
      config,
      train_data,
      save_folder=save_folder,
      logs_folder=logs_folder)


if __name__ == "__main__":
  main()
