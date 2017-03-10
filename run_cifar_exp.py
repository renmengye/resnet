#!/usr/bin/env python
"""
Train a CNN on CIFAR.
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

import numpy as np
import os
import tensorflow as tf

from tqdm import tqdm

from resnet.configs.imagenet_exp_config import get_config, get_config_from_json
from resnet.data import get_dataset
from resnet.models import ResNetModel
from resnet.utils import ExperimentLogger, FixedLearnRateScheduler
from resnet.utils import logger, gen_id

log = logger.get()

flags = tf.flags
flags.DEFINE_string("config", None, "Manually defined config file.")
flags.DEFINE_string("dataset", "cifar-10", "Dataset name.")
flags.DEFINE_string("id", None, "Experiment ID.")
flags.DEFINE_string("results", "./results/cifar", "Saving folder.")
flags.DEFINE_string("logs", "./logs/public", "Logging folder.")
flags.DEFINE_string("model", "resnet-32", "Model type.")
flags.DEFINE_bool("validation", False, "Whether run validation set.")
FLAGS = flags.FLAGS


def _get_config():
  # Manually set config.
  if FLAGS.config is not None:
    return get_config_from_json(FLAGS.config)
  else:
    return get_config(FLAGS.dataset, FLAGS.model)


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


def get_models(config):
  # Builds models.
  log.info("Building models")
  with tf.name_scope("Train"):
    with tf.variable_scope("Model", reuse=None):
      m = ResNetModel(config, is_training=True)

  with tf.name_scope("Valid"):
    with tf.variable_scope("Model", reuse=True):
      mvalid = ResNetModel(config, is_training=False)
  return m, mvalid


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
  log.info("Config: {}".format(config.__dict__))
  exp_logger = ExperimentLogger(logs_folder)

  # Initializes variables.
  with tf.Graph().as_default():
    np.random.seed(0)
    if not hasattr(config, "seed"):
      tf.set_random_seed(1234)
      log.info("Setting tensorflow random seed={:d}".format(1234))
    else:
      log.info("Setting tensorflow random seed={:d}".format(config.seed))
      tf.set_random_seed(config.seed)
    m, mvalid = get_models(config)

    with tf.Session() as sess:
      saver = tf.train.Saver()
      sess.run(tf.global_variables_initializer())

      # Set up learning rate schedule.
      if config.lr_scheduler_type == "fixed":
        lr_scheduler = FixedLearnRateScheduler(
            sess,
            m,
            config.base_learn_rate,
            config.lr_decay_steps,
            lr_list=config.lr_list)
      elif config.lr_scheduler_type == "exponential":
        lr_scheduler = ExponentialLearnRateScheduler(
            sess, m, config.base_learn_rate, config.lr_decay_offset,
            config.max_train_iter, config.final_learn_rate,
            config.lr_decay_interval)
      else:
        raise Exception("Unknown learning rate scheduler {}".format(
            config.lr_scheduler))

      for niter in tqdm(range(config.max_train_iter), desc=exp_id):
        lr_scheduler.step(niter)
        ce = train_step(sess, m, train_iter.next())

        if (niter + 1) % config.disp_iter == 0 or niter == 0:
          exp_logger.log_train_ce(niter, ce)

        if (niter + 1) % config.valid_iter == 0 or niter == 0:
          if trainval_iter is not None:
            trainval_iter.reset()
            acc = evaluate(sess, mvalid, trainval_iter)
            exp_logger.log_train_acc(niter, acc)
          test_iter.reset()
          acc = evaluate(sess, mvalid, test_iter)
          exp_logger.log_valid_acc(niter, acc)

        if (niter + 1) % config.save_iter == 0 or niter == 0:
          save(sess, saver, m.global_step, config, save_folder)
          exp_logger.log_learn_rate(niter, m.lr.eval())

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
    exp_id = "exp_" + FLAGS.dataset + "_" + FLAGS.model
    exp_id = gen_id(exp_id)
  else:
    exp_id = FLAGS.id

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
  train_data = get_dataset(FLAGS.dataset, train_str)
  trainval_data = get_dataset(
      FLAGS.dataset,
      train_str,
      num_batches=100,
      data_aug=False,
      cycle=False,
      prefetch=False)
  test_data = get_dataset(
      FLAGS.dataset, test_str, data_aug=False, cycle=False, prefetch=False)

  # Trains a model.
  acc = train_model(
      exp_id,
      config,
      train_data,
      test_data,
      trainval_data,
      save_folder=save_folder,
      logs_folder=logs_folder)
  log.info("Final test accuracy = {:.3f}".format(acc * 100))


if __name__ == "__main__":
  main()
