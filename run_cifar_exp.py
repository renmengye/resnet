#!/usr/bin/env python
"""
Author: Mengye Ren (mren@cs.toronto.edu)

Usage:
python run_cifar_exp.py    --model           [MODEL NAME]          \
                           --config          [CONFIG FILE]         \
                           --env             [ENV FILE]            \
                           --dataset         [DATASET]             \
                           --data_folder     [DATASET FOLDER]      \
                           --validation                            \
                           --no_validation                         \
                           --logs            [LOGS FOLDER]         \
                           --results         [SAVE FOLDER]         \
                           --gpu             [GPU ID]              

Flags:
  --model: See resnet/configs/cifar_exp_config.py. Default resnet-32.
  --config: Not using the pre-defined configs above, specify the JSON file
  that contains model configurations.
  --dataset: Dataset name. Available options are: 1) cifar-10 2) cifar-100.
  --data_folder: Path to data folder, default is data/{DATASET}.
  --validation: Evaluating experiments on validation set.
  --no_validation: Evaluating experiments on test set.
  --logs: Path to logs folder, default is logs/default.
  --results: Path to save folder, default is results.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import json
import numpy as np
import os
import sys
import tensorflow as tf

from tqdm import tqdm

from resnet.configs import cifar_exp_config as conf
from resnet.data import get_dataset
from resnet.models import ResNetModel
from resnet.utils import ExperimentLogger
from resnet.utils import logger, gen_id

log = logger.get()

flags = tf.flags
flags.DEFINE_string("config", None, "Manually defined config file.")
flags.DEFINE_string("dataset", "cifar-10", "Dataset name.")
flags.DEFINE_string("id", None, "Experiment ID.")
flags.DEFINE_string("results", "results/cifar", "Saving folder.")
flags.DEFINE_string("logs", "logs/default", "Logging folder.")
flags.DEFINE_string("model", "resnet-32", "Model type.")
flags.DEFINE_bool("validation", False, "Whether run validation set.")
FLAGS = flags.FLAGS


def get_config():
  # Manually set config.
  if FLAGS.config is not None:
    return conf.get_config_from_json(FLAGS.config)
  else:
    return conf.get_config(FLAGS.dataset, FLAGS.model)


def train_step(sess, model, batch):
  """Train step."""
  feed_data = {model.input: batch["img"], model.label: batch["label"]}
  cost, ce, _ = sess.run([model.cost, model.cross_ent, model.train_op],
                         feed_dict=feed_data)
  return ce


def evaluate(sess, model, data_iter):
  """Runs evaluation."""
  num_correct = 0.0
  count = 0
  for batch in data_iter:
    feed_data = {model.input: batch["img"]}
    y = sess.run(model.output, feed_dict=feed_data)
    pred_label = np.argmax(y, axis=1)
    num_correct += np.sum(np.equal(pred_label, batch["label"]).astype(float))
    count += pred_label.size
  acc = (num_correct / count)
  return acc


def save(sess, saver, global_step, config, save_folder):
  """Snapshots a model."""
  if not os.path.isdir(save_folder):
    os.makedirs(save_folder)
    config_file = os.path.join(save_folder, "conf.json")
    with open(config_file, "w") as f:
      f.write(config.to_json())
  log.info("Saving to {}".format(save_folder))
  saver.save(
      sess, os.path.join(save_folder, "model.ckpt"), global_step=global_step)


def train_model(exp_id,
                config,
                train_iter,
                test_iter,
                trainval_iter=None,
                save_folder=None,
                logs_folder=None):
  """Trains a CIFAR model.

  Args:
      exp_id: String. Experiment ID.
      config: Config object
      train_data: Dataset iterator.
      test_data: Dataset iterator.

  Returns:
      acc: Final test accuracy
  """
  np.random.seed(0)
  if not hasattr(config, "seed"):
    tf.set_random_seed(1234)
    log.info("Setting tensorflow random seed={:d}".format(1234))
  else:
    log.info("Setting tensorflow random seed={:d}".format(config.seed))
    tf.set_random_seed(config.seed)

  log.info("Config: {}".format(config.__dict__))
  if save_folder is not None:
    save_folder = os.path.join(save_folder, exp_id)
  if logs_folder is not None:
    logs_folder = os.path.join(logs_folder, exp_id)
  exp_logger = ExperimentLogger(logs_folder)

  # Builds models.
  log.info("Building models")
  with tf.name_scope("Train"):
    with tf.variable_scope("Model", reuse=None):
      m = ResNetModel(config, is_training=True)

  with tf.name_scope("Valid"):
    with tf.variable_scope("Model", reuse=True):
      mvalid = ResNetModel(config, is_training=False)

  # Initializes variables.
  with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    lr = config.base_learn_rate
    lr_decay_steps = config.lr_decay_steps
    max_train_iter = config.max_train_iter
    m.assign_lr(sess, lr)
    for niter in tqdm(range(max_train_iter)):
      # Decrease learning rate.
      if len(lr_decay_steps) > 0:
        if (niter + 1) == lr_decay_steps[0]:
          lr *= 0.1
          m.assign_lr(sess, lr)
          lr_decay_steps.pop(0)

      ce = train_step(sess, m, train_iter.next())

      if (niter + 1) % config.disp_iter == 0 or niter == 0:
        exp_logger.log_train_ce(niter, ce)

      if (niter + 1) % config.valid_iter == 0 or niter == 0:
        if trainval_iter is not None:
          acc = evaluate(sess, mvalid, trainval_iter)
          exp_logger.log_train_acc(niter, acc)
        test_iter.reset()
        acc = evaluate(sess, mvalid, test_iter)
        log.info("Experment ID {}".format(exp_id))
        exp_logger.log_valid_acc(niter, acc)

      if (niter + 1) % config.save_iter == 0:
        if save_folder is not None:
          save(sess, saver, m.global_step, config, save_folder)
    test_iter.reset()
    acc = evaluate(test_iter, -1)
  return acc


def main():
  # Loads parammeters.
  config = get_config()

  if FLAGS.validation:
    train_str = "traintrain"
    test_str = "trainval"
    log.warning("Running validation set")
  else:
    train_str = "train"
    test_str = "test"

  if FLAGS.id is None:
    exp_id = "exp_cifar" + FLAGS.dataset + "_" + FLAGS.model
    exp_id = gen_id(exp_id)

  # Configures dataset objects.
  log.info("Building dataset")
  train_data = get_dataset(FLAGS.dataset, train_str)
  trainval_data = get_dataset(
      FLAGS.dataset, train_str, num_batches=100, cycle=False)
  test_data = get_dataset(FLAGS.dataset, test_str, cycle=False)

  # Trains a model.
  acc = train_model(exp_id, config, train_data, test_data, trainval_data)
  log.info("Final test accuracy = {:.3f}".format(acc * 100))


if __name__ == "__main__":
  main()
