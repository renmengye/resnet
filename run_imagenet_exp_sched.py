#!/usr/bin/env python
"""
Launches ImageNet experiment for a limited number of steps, periodically.
Author: Mengye Ren (mren@cs.toronto.edu)

Usage:
./run_imagenet_exp_sched.py --id              [EXPERIMENT ID]          \
                            --logs            [LOGS FOLDER]            \
                            --results         [SAVE FOLDER]            \  
                            --max_num_steps   [MAX NUMBER OF STEPS]    \
                            --max_max_steps   [TOTAL NUMBER OF STEPS]  \
                            --model           [MODEL NAME]
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import sys
import tensorflow as tf
import time
import traceback

from pysched.slurm import SlurmCommandDispatcherFactory
from pysched.local import LocalCommandDispatcherFactory
from resnet.utils import gen_id, logger

log = logger.get()

flags = tf.flags
flags.DEFINE_string("id", None, "Experiment ID")
flags.DEFINE_string("results", "./results/imagenet", "Saving folder")
flags.DEFINE_string("logs", "./logs/public", "Logging folder")
flags.DEFINE_bool("local", False, "Whether run locally or on slurm")
flags.DEFINE_integer("max_num_steps", 30000, "Maximum number of steps")
flags.DEFINE_integer("max_max_steps", 600000,
                     "Maximum number of training steps")
flags.DEFINE_integer("num_pass", 1, "Number of forward-backwad passes")
flags.DEFINE_integer("min_interval", 7200, "Minimum number of seconds")
flags.DEFINE_string("model", "resnet-50", "Model name")
flags.DEFINE_string("machine", None, "Preferred machine")
FLAGS = flags.FLAGS
DATASET = "imagenet"

# Get dispatcher factory.
if FLAGS.local:
  dispatch_factory = LocalCommandDispatcherFactory()
else:
  dispatch_factory = SlurmCommandDispatcherFactory()

# Generate experiment ID.
if FLAGS.id is None:
  exp_id = gen_id("exp_" + DATASET + "_" + FLAGS.model)
  restore = False
  # raise Exception("You need to specify model ID.")
else:
  exp_id = FLAGS.id
  restore = True

save_folder = os.path.realpath(
    os.path.abspath(os.path.join(FLAGS.results, exp_id)))

while True:

  # Check if we need to launch another job.
  if os.path.exists(save_folder):
    latest_ckpt = tf.train.latest_checkpoint(save_folder)
    cur_steps = int(latest_ckpt.split("-")[-1])
  else:
    cur_steps = 0
  if cur_steps >= FLAGS.max_max_steps:
    log.info("Maximum steps {} reached.".format(FLAGS.max_max_steps))
    break

  # Use slurm to launch job.
  try:
    start_time = time.time()
    log.info("Training model \"{}\"".format(exp_id))
    dispatcher = dispatch_factory.create(
        num_gpu=4, num_cpu=12, machine=FLAGS.machine)
    arg_list = [
        "./run_imagenet_exp.py", "--id", exp_id, "--results", FLAGS.results,
        "--logs", FLAGS.logs, "--max_num_steps", str(FLAGS.max_num_steps),
        "--model", FLAGS.model, "--num_pass", str(FLAGS.num_pass), "--verbose",
        "--num_gpu", "4"
    ]
    if restore:
      arg_list.append("--restore")
    restore = True  # Restore at the next time!
    job = dispatcher.dispatch(arg_list)
    code = job.wait()
    if code != 0:
      log.error("Job failed")
  except Exception as e:
    log.error("An exception occurred.")
    log.error(e)
    exc_type, exc_value, exc_traceback = sys.exc_info()
    log.error("*** print_tb:")
    traceback.print_tb(exc_traceback, limit=10, file=sys.stdout)
    log.error("*** print_exception:")
    traceback.print_exception(
        exc_type, exc_value, exc_traceback, limit=10, file=sys.stdout)

  # Wait for the next job.
  end_time = time.time()
  elapsed = end_time - start_time
  if elapsed < FLAGS.min_interval:
    time.sleep(FLAGS.min_interval - elapsed)
