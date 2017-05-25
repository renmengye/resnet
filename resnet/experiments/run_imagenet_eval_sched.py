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

import os
import sys
import tensorflow as tf
import time
import traceback

from pyschedlib.slurm import SlurmCommandDispatcherFactory
from pyschedlib.local import LocalCommandDispatcherFactory
from resnet.utils import logger

log = logger.get()

flags = tf.flags
flags.DEFINE_string("id", None, "Experiment ID")
flags.DEFINE_string("machine", None, "Preferred machine")
flags.DEFINE_string("results", "./results/imagenet", "Saving folder")
flags.DEFINE_string("logs", "./logs/public", "Logging folder")
flags.DEFINE_bool("local", False, "Whether run locally or on slurm")
flags.DEFINE_integer("min_interval", 7200, "Minimum number of seconds")
flags.DEFINE_bool("sweep_all_ckpt", False, "Whether to sweep all checkpoints")
FLAGS = flags.FLAGS


def scan_folder(folder):
  files = os.listdir(folder)
  numbers = set()
  for ff in files:
    if "ckpt" not in ff:
      continue
    parts = ff.split(".")
    num = parts[1].split("-")[1]
    numbers.add(int(num))
  return list(numbers)


def main():
  ckpt_ran = []
  if FLAGS.local:
    dispatch_factory = LocalCommandDispatcherFactory()
  else:
    dispatch_factory = SlurmCommandDispatcherFactory()

  if FLAGS.id is None:
    raise Exception("You need to specify model ID.")

  if FLAGS.sweep_all_ckpt:
    ckpt_num = 0
  else:
    ckpt_num = -1
  while True:
    try:
      start_time = time.time()
      ckpt_list = scan_folder("results/imagenet/" + FLAGS.id)
      ckpt_list = list(sorted(ckpt_list))
      if FLAGS.sweep_all_ckpt:
        found = False
        # Find a checkpoint that is not run.
        for ckpt_num in ckpt_list:
          if ckpt_num not in ckpt_ran:
            found = True
            break
      else:
        found = True
      if found:
        log.info("Evaluating model \"{}\" at step {}".format(FLAGS.id,
                                                             ckpt_num))
        if FLAGS.local:
          dispatcher = dispatch_factory.create()
        else:
          dispatcher = dispatch_factory.create(
              num_gpu=1, num_cpu=2, machine=FLAGS.machine)
        job = dispatcher.dispatch([
            "./run_imagenet_eval.py", "--id", FLAGS.id, "--results",
            FLAGS.results, "--logs", FLAGS.logs, "--ckpt_num", str(ckpt_num)
        ])
        code = job.wait()
        if code != 0:
          log.error("Job failed")
          log.error("Will try re-running {}".format(ckpt_num))
        else:
          if FLAGS.sweep_all_ckpt:
            ckpt_ran.append(ckpt_num)
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


if __name__ == "__main__":
  main()
