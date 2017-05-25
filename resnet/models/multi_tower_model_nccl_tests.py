"""Unit tests for multi-tower model."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
import tensorflow as tf
from resnet.configs import test_configs
from resnet.configs import get_config
from resnet.models import get_model, get_multi_gpu_model
from resnet.models.rev_resnet_model_tests import check_two_dict
from resnet.utils import logger

log = logger.get()
FOLDER = "tmp"
CKPT_FNAME = os.path.join(FOLDER, "test.ckpt")


class MultiTowerModelNCCLTests(tf.test.TestCase):
  """Unit tests for MultiTowerModelNCCL. Make sure you run them on GPU."""

  def test_fw(self):
    """Tests the forward computation is the same."""
    with tf.Graph().as_default(), tf.Session() as sess, log.verbose_level(2):
      config = get_config("test", "resnet-test")
      config.num_channel = 4
      config.height = 8
      config.width = 8
      np.random.seed(0)
      xval = np.random.uniform(-1.0, 1.0, [10, 8, 8, 4]).astype(np.float32)
      x = tf.constant(xval)
      x1 = x[:5, :, :, :]
      x2 = x[5:, :, :, :]
      # We need to split two regular runs because of the complication brought by
      # batch normalization.
      with tf.variable_scope("Model", reuse=None):
        m11 = get_model("resnet", config, inp=x1)
      with tf.variable_scope("Model", reuse=True):
        m12 = get_model("resnet", config, inp=x2)
      with tf.variable_scope("Model2", reuse=None):
        m2 = get_multi_gpu_model(
            "resnet-gpu", config, num_replica=2, inp=x, use_nccl=True)
      sess.run(tf.global_variables_initializer())
      tvars = tf.global_variables()
      tvars_str = map(lambda x: x.name, tvars)
      tvars_val = sess.run(tvars)
      tvars_d = dict(zip(tvars_str, tvars_val))

      aop = []
      for vv in tvars:
        if vv.name.startswith("Model2/replica_"):
          aop.append(
              tf.assign(vv, tvars_d[vv.name.replace(
                  "Model2/replica_0/", "Model/").replace("Model2/replica_1/",
                                                         "Model/")]))
      sess.run(aop)
      y11, y12, y2 = sess.run([m11.output, m12.output, m2.output])
      np.testing.assert_allclose(y11, y2[:5, :], rtol=1e-5)
      np.testing.assert_allclose(y12, y2[5:, :], rtol=1e-5)

  def test_bk(self):
    """Tests the backward computation is the same."""
    with tf.Graph().as_default(), tf.Session() as sess, log.verbose_level(2):
      config = get_config("test", "resnet-test")
      config.num_channel = 4
      config.height = 8
      config.width = 8
      np.random.seed(0)
      xval = np.random.uniform(-1.0, 1.0, [10, 8, 8, 4]).astype(np.float32)
      yval = np.floor(np.random.uniform(0, 9.9, [10])).astype(np.int32)
      x = tf.constant(xval)
      y = tf.constant(yval)
      with tf.variable_scope("Model", reuse=None):
        m1 = get_multi_gpu_model(
            "resnet", config, num_replica=2, inp=x, label=y)
      with tf.variable_scope("Model2", reuse=None):
        m2 = get_multi_gpu_model(
            "resnet-gpu", config, num_replica=2, inp=x, label=y, use_nccl=True)

      sess.run(tf.global_variables_initializer())
      m1.assign_lr(sess, 0.1)
      m2.assign_lr(sess, 0.1)
      tvars = tf.trainable_variables()
      saver = tf.train.Saver()
      saver.save(sess, CKPT_FNAME)
      tvars_str = map(lambda x: x.name, tvars)
      tvars_val = sess.run(tvars)
      tvars_d = dict(zip(tvars_str, tvars_val))
      tvars_dv = dict(zip(tvars_str, tvars))

      convert_name = lambda x: x.replace("Model2/replica_0/", "Model/").replace("Model2/replica_1/", "Model/")
      aop = []
      for vv in tvars:
        if vv.name.startswith("Model2/replica_"):
          aop.append(tf.assign(vv, tvars_d[convert_name(vv.name)]))
      sess.run(aop)

      m1vars = []
      m2vars = []
      for vv in tvars:
        if vv.name.startswith("Model2/replica_0"):
          m2vars.append(vv)
          m1vars.append(tvars_dv[convert_name(vv.name)])

      m1.train_step(sess)
      m1val = sess.run(m1vars)
      m1d = dict(zip(map(lambda x: x.name, m1vars), m1val))

      saver.restore(sess, CKPT_FNAME)
      m2.train_step(sess)
      m2val = sess.run(m2vars)
      m2d = dict(zip(map(lambda x: convert_name(x.name), m2vars), m2val))

      # Check the average gradients are the same.
      check_two_dict(m1d, m2d)


if __name__ == "__main__":
  tf.test.main()
