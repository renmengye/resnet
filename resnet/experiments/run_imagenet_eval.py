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

from resnet.configs.config_factory import get_config_from_json
from resnet.data import get_dataset
from resnet.models import get_model
from resnet.utils import logger, ExperimentLogger

flags = tf.flags
flags.DEFINE_string("id", None, "Experiment ID")
flags.DEFINE_string("results", "./results/imagenet", "Saving folder")
flags.DEFINE_string("logs", "./logs/public", "Logging folder")
flags.DEFINE_integer("ckpt_num", -1, "Checkpoint step number")
FLAGS = tf.flags.FLAGS
log = logger.get()


def _get_config():
  save_folder = os.path.join(FLAGS.results, FLAGS.id)
  return get_config_from_json(os.path.join(save_folder, "conf.json"))


def _get_model(config):
  with log.verbose_level(2):
    with tf.name_scope("Valid"):
      with tf.variable_scope("Model"):
        log.info(config.name)
        mvalid = get_model(
            config.model_class, config, is_training=False, inference_only=True)
  return mvalid


def evaluate(sess, model, data_iter):
  """Runs evaluation."""
  num_correct = 0.0
  count = 0
  iter_ = tqdm(data_iter)
  for batch in iter_:
    y = model.infer_step(sess, batch["img"])
    pred_label = np.argmax(y, axis=1)
    # print(np.concatenate(pred_label, batch["label"], axis=np.newaxis))
    num_correct += np.sum(np.equal(pred_label, batch["label"]).astype(float))
    count += pred_label.size
  acc = (num_correct / count)
  return acc


def eval_model(config,
               train_data,
               test_data,
               save_folder,
               logs_folder=None,
               ckpt_num=-1):
  log.info("Config: {}".format(config.__dict__))

  with tf.Graph().as_default():
    np.random.seed(0)
    tf.set_random_seed(1234)
    exp_logger = ExperimentLogger(logs_folder)

    # Builds models.
    log.info("Building models")
    mvalid = _get_model(config)

    # # A hack to load compatible models.
    # variables = tf.global_variables()
    # names = map(lambda x: x.name, variables)
    # names = map(lambda x: x.replace("Model/", "Model/Towers/"), names)
    # names = map(lambda x: x.replace(":0", ""), names)
    # var_dict = dict(zip(names, variables))

    # Initializes variables.
    with tf.Session() as sess:
      # saver = tf.train.Saver(var_dict)
      saver = tf.train.Saver()
      if ckpt_num == -1:
        ckpt = tf.train.latest_checkpoint(save_folder)
      elif ckpt_num >= 0:
        ckpt = os.path.join(save_folder, "model.ckpt-{}".format(ckpt_num))
      else:
        raise ValueError("Invalid checkpoint number {}".format(ckpt_num))
      if not os.path.exists(ckpt + ".meta"):
        raise ValueError("Checkpoint not exists")
      saver.restore(sess, ckpt)
      train_acc = evaluate(sess, mvalid, train_data)
      val_acc = evaluate(sess, mvalid, test_data)
      niter = int(ckpt.split("-")[-1])
      exp_logger.log_train_acc(niter, train_acc)
      exp_logger.log_valid_acc(niter, val_acc)
    return val_acc


def main():
  config = _get_config()
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
      #num_batches=5,
      preprocessor=config.preprocessor)
  test_data = get_dataset(
      "imagenet",
      "valid",
      cycle=False,
      data_aug=False,
      batch_size=config.valid_batch_size,
      #num_batches=5,
      preprocessor=config.preprocessor)

  # Evaluates a model.
  eval_model(
      config,
      train_data,
      test_data,
      save_folder,
      logs_folder,
      ckpt_num=FLAGS.ckpt_num)


if __name__ == "__main__":
  main()
