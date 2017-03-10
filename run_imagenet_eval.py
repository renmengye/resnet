#!/usr/bin/env python
"""
Evaluates a CNN on ImageNet.
Author: Mengye Ren (mren@cs.toronto.edu)

Usage:
./run_imagenet_eval.py --id              [EXPERIMENT ID]     \
                       --logs            [LOGS FOLDER]       \
                       --results         [SAVE FOLDER]       

Flags:
  --id: Experiment ID, optional for new experiment.
  --logs: Path to logs folder, default is ./logs/default.
  --results: Path to save folder, default is ./results/imagenet.
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
from resnet.utils import logger, ExperimentLogger

flags = tf.flags
flags.DEFINE_string("id", None, "eExperiment ID")
flags.DEFINE_string("results", "./results/imagenet", "Saving folder")
flags.DEFINE_string("logs", "./logs/default", "Logging folder")
FLAGS = tf.flags.FLAGS
log = logger.get()


def get_config():
  save_folder = os.path.join(FLAGS.results, FLAGS.id)
  return get_config_from_json(os.path.join(save_folder, "conf.json"))


def get_model(config):
  with tf.name_scope("Valid"):
    with tf.variable_scope("Model"):
      mvalid = ResNetModel(config, is_training=False)
  return mvalid


def evaluate(sess, model, data_iter):
  """Runs evaluation."""
  num_correct = 0.0
  count = 0
  iter_ = tqdm(data_iter)
  for batch in iter_:
    y = model.infer_step(sess, batch["img"])
    pred_label = np.argmax(y, axis=1)
    num_correct += np.sum(np.equal(pred_label, batch["label"]).astype(float))
    count += pred_label.size
  acc = (num_correct / count)
  return acc


def eval_model(config, train_data, test_data, save_folder, logs_folder=None):
  log.info("Config: {}".format(config.__dict__))

  with tf.Graph().as_default():
    np.random.seed(0)
    tf.set_random_seed(1234)
    exp_logger = ExperimentLogger(logs_folder)

    # Builds models.
    log.info("Building models")
    mvalid = get_model(config)

    # Initializes variables.
    with tf.Session() as sess:
      saver = tf.train.Saver()
      ckpt = tf.train.latest_checkpoint(save_folder)
      saver.restore(sess, ckpt)
      train_acc = evaluate(sess, mvalid, train_data)
      val_acc = evaluate(sess, mvalid, test_data)
      niter = int(ckpt.split("-")[-1])
      exp_logger.log_train_acc(niter, train_acc)
      exp_logger.log_valid_acc(niter, val_acc)
    return val_acc


def main():
  config = get_config()
  exp_id = FLAGS.id

  save_folder = os.path.realpath(
      os.path.abspath(os.path.join(FLAGS.results, exp_id)))

  if FLAGS.logs is not None:
    logs_folder = os.path.realpath(
        os.path.abspath(os.path.join(FLAGS.logs, exp_id)))
    if not os.path.exists(logs_folder):
      os.makedirs(logs_folder)
  else:
    logs_folder = None

  # Configures dataset objects.
  log.info("Building dataset")
  train_data = get_dataset(
      "imagenet",
      "train",
      cycle=False,
      data_aug=False,
      batch_size=config.valid_batch_size,
      num_batches=100,
      preprocessor=config.preprocessor)
  test_data = get_dataset(
      "imagenet",
      "valid",
      cycle=False,
      data_aug=False,
      batch_size=config.valid_batch_size,
      preprocessor=config.preprocessor)

  # Evaluates a model.
  eval_model(config, train_data, test_data, save_folder, logs_folder)


if __name__ == "__main__":
  main()
