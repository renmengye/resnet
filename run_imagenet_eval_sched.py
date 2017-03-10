#!/usr/bin/env python
"""
Evaluates ImageNet experiment, periodically.
Author: Mengye Ren (mren@cs.toronto.edu)

Usage:
./run_imagenet_eval_sched.py --id         [EXPERIMENT ID]               \
                            --results     [SAVE FOLDER]                 \
                            --local       [LAUNCH LOCALLY OR SLURM]     \
                            --machine     [PREFERRED SLURM MACHINE NAME]
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import tensorflow as tf
import time
import traceback

from pysched.slurm import SlurmCommandDispatcherFactory
from pysched.local import LocalCommandDispatcherFactory
from resnet.utils import logger

log = logger.get()

flags = tf.flags
flags.DEFINE_string("id", None, "Experiment ID")
flags.DEFINE_string("machine", None, "Preferred machine")
flags.DEFINE_string("results", "./results/imagenet", "Saving folder")
flags.DEFINE_string("logs", "./logs/default", "Logging folder")
flags.DEFINE_bool("local", False, "Whether run locally or on slurm")
flags.DEFINE_integer("min_interval", 7200, "Minimum number of seconds")
FLAGS = flags.FLAGS

if FLAGS.local:
  dispatch_factory = LocalCommandDispatcherFactory()
else:
  dispatch_factory = SlurmCommandDispatcherFactory()

if FLAGS.id is None:
  raise Exception("You need to specify model ID.")
while True:
  try:
    start_time = time.time()
    log.info("Evaluating model \"{}\"".format(FLAGS.id))
    dispatcher = dispatch_factory.create(
        num_gpu=1, num_cpu=2, machine=FLAGS.machine)
    job = dispatcher.dispatch([
        "./run_imagenet_eval.py", "--id", FLAGS.id, "--results", FLAGS.results,
        "--logs", FLAGS.logs
    ])
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
  end_time = time.time()
  elapsed = end_time - start_time
  if elapsed < FLAGS.min_interval:
    time.sleep(FLAGS.min_interval - elapsed)
