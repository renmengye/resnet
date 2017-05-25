"""Unit tests for multi-pass model."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
import tensorflow as tf
from resnet.configs import test_configs
from resnet.configs import get_config
from resnet.models import get_model, get_multi_gpu_model
from resnet.models.multi_pass_model_v2 import MultiPassModelV2
from resnet.models.resnet_model import ResNetModel
from resnet.utils import logger

log = logger.get()
FOLDER = "tmp"
CKPT_FNAME = os.path.join(FOLDER, "test.ckpt")


class MultiPassModelTests(tf.test.TestCase):

  def test_single_pass(self):
    """Tests multi-pass is the same with multi-tower."""
    for method in ["cumsum", "storage"]:
      with tf.Graph().as_default(), tf.Session() as sess, log.verbose_level(2):
        config = get_config("test", "resnet-test")
        config.momentum = 0.0
        config.base_learn_rate = 1e-1
        config.num_channel = 4
        config.height = 8
        config.width = 8
        np.random.seed(0)
        BSIZE = 100
        xval = np.random.uniform(
            -1.0, 1.0, [BSIZE, config.height, config.width,
                        config.num_channel]).astype(np.float32)
        yval = np.floor(np.random.uniform(0, 9.9, [BSIZE])).astype(np.int32)
        x = tf.constant(xval)
        y = tf.constant(yval)
        with tf.variable_scope("Model", reuse=None):
          m1 = get_multi_gpu_model(
              "resnet", config, num_replica=2, inp=x, label=y)
        with tf.variable_scope("Model", reuse=True):
          m = get_model("resnet", config, inp=x, label=y)
        with tf.variable_scope("Model", reuse=True):
          m2 = MultiPassModelV2(
              config,
              ResNetModel,
              num_passes=2,
              debug=True,
              inp=x,
              label=y,
              aggregate_method=method)
        sess.run(tf.global_variables_initializer())
        m2.assign_lr(sess, config.base_learn_rate)
        tvars = tf.trainable_variables()
        tvars_str = map(lambda x: x.name, tvars)
        tvars_v0 = sess.run(tvars)
        tvars_d0 = dict(zip(tvars_str, tvars_v0))

        # Run multi tower version.
        saver = tf.train.Saver()
        saver.save(sess, CKPT_FNAME)
        m1.train_step(sess)
        tvars_v1 = sess.run(tvars)
        tvars_d1 = dict(zip(tvars_str, tvars_v1))

        # Original ResNet.
        saver.restore(sess, CKPT_FNAME)
        m.train_step(sess)
        tvars_v = sess.run(tvars)
        tvars_d = dict(zip(tvars_str, tvars_v))

        # Run multi tower version gradients.
        saver.restore(sess, CKPT_FNAME)
        namelist1 = map(lambda x: x[1].name, m1.grads_and_vars)
        grads1 = map(lambda x: x[0], m1.grads_and_vars)
        gradsval1 = sess.run(grads1)
        gdict1 = dict(zip(namelist1, gradsval1))

        # Get only the first tower of multi tower.
        saver.restore(sess, CKPT_FNAME)
        namelist11 = map(lambda x: x[1].name, m1.tower_grads_and_vars[0])
        grads11 = map(lambda x: x[0], m1.tower_grads_and_vars[0])
        gradsval11 = sess.run(grads11)
        gdict11 = dict(zip(namelist11, gradsval11))

        # Get only the first tower of multi tower.
        saver.restore(sess, CKPT_FNAME)
        namelist12 = map(lambda x: x[1].name, m1.tower_grads_and_vars[1])
        grads12 = map(lambda x: x[0], m1.tower_grads_and_vars[1])
        gradsval12 = sess.run(grads12)
        gdict12 = dict(zip(namelist12, gradsval12))

        # Run MultiPassModel.
        saver.restore(sess, CKPT_FNAME)
        m2.train_step(sess)
        tvars_v2 = sess.run(tvars)
        tvars_d2 = dict(zip(tvars_str, tvars_v2))

        # Get grads and vars in multi pass mode by a hack in debug mode.
        grads = map(lambda x: m2.optimizer.grad_cache[x], tvars)
        m2_grads_and_vars = zip(grads, tvars)

        #saver.restore(sess, CKPT_FNAME)
        namelist2 = map(lambda x: x[1].name, m2_grads_and_vars)
        grads2 = map(lambda x: x[0], m2_grads_and_vars)
        gradsval2 = sess.run(grads2)
        gdict2 = dict(zip(namelist2, gradsval2))

        # Get only the first pass of multi pass.
        saver.restore(sess, CKPT_FNAME)
        namelist21 = map(lambda x: x[1].name, m2.model.grads_and_vars)
        grads21 = map(lambda x: x[0], m2.model.grads_and_vars)
        gradsval21 = sess.run(grads21, {m2._pass_id: 0})
        gdict21 = dict(zip(namelist21, gradsval21))

        # Get only the second pass of multi pass.
        saver.restore(sess, CKPT_FNAME)
        namelist22 = map(lambda x: x[1].name, m2.model.grads_and_vars)
        grads22 = map(lambda x: x[0], m2.model.grads_and_vars)
        gradsval22 = sess.run(grads22, {m2._pass_id: 1})
        gdict22 = dict(zip(namelist22, gradsval22))

        # Get only the first pass of multi pass on the variable.
        saver.restore(sess, CKPT_FNAME)
        sess.run(m2.train_op_list[0], {m2._pass_id: 0})
        gradsval21c = sess.run(grads2)
        gdict21c = dict(zip(namelist2, gradsval21c))

        # Get only the first pass of multi pass on the variable.
        saver.restore(sess, CKPT_FNAME)
        sess.run(m2.train_op_list[1], {m2._pass_id: 1})
        gradsval22c = sess.run(grads2)
        gdict22c = dict(zip(namelist2, gradsval22c))

        # Get both passes.
        saver.restore(sess, CKPT_FNAME)
        sess.run(m2.train_op_list[0], {m2._pass_id: 0})
        sess.run(m2.train_op_list[1], {m2._pass_id: 1})
        gradsval2c = sess.run(grads2)
        gdict2c = dict(zip(namelist2, gradsval2c))

        for vv in gdict1.keys():
          log.info(vv, verbose=2)
          np.testing.assert_allclose(
              tvars_d1[vv],
              tvars_d0[vv] - config.base_learn_rate * gdict1[vv],
              rtol=1e-4,
              atol=1e-6)
          np.testing.assert_allclose(
              tvars_d[vv], tvars_d1[vv], rtol=1e-1, atol=1e-2)
          np.testing.assert_allclose(
              gdict11[vv], gdict21[vv], rtol=1e-4, atol=1e-6)
          np.testing.assert_allclose(
              gdict12[vv], gdict22[vv], rtol=1e-4, atol=1e-6)

          if m2.optimizer._method == "storage":
            np.testing.assert_allclose(
                gdict11[vv], gdict21c[vv][0], rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(
                gdict12[vv], gdict22c[vv][1], rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(
                gdict11[vv], gdict2[vv][0], rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(
                gdict12[vv], gdict2[vv][1], rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(
                gdict11[vv], gdict2c[vv][0], rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(
                gdict12[vv], gdict2c[vv][1], rtol=1e-4, atol=1e-6)
          elif m2.optimizer._method == "cumsum":
            np.testing.assert_allclose(
                gdict11[vv], gdict21c[vv] * 2, rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(
                gdict12[vv], gdict22c[vv] * 2, rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(
                gdict21c[vv] + gdict22c[vv], gdict2[vv], rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(
                gdict2c[vv] - gdict21c[vv], gdict22c[vv], rtol=1e-4, atol=1e-6)
          np.testing.assert_allclose(
              tvars_d1[vv], tvars_d2[vv], rtol=1e-4, atol=1e-6)
          log.info("...ok", verbose=2)

  def test_multi_pass(self):
    """Tests multi-pass is the same with multi-tower."""
    for method in ["cumsum", "storage"]:
      with tf.Graph().as_default(), tf.Session() as sess, log.verbose_level(2):
        config = get_config("test", "resnet-test")
        config.momentum = 0.0
        config.base_learn_rate = 1e-1
        config.num_channel = 4
        config.height = 8
        config.width = 8
        np.random.seed(0)
        BSIZE = 100
        xval = np.random.uniform(
            -1.0, 1.0, [BSIZE, config.height, config.width,
                        config.num_channel]).astype(np.float32)
        yval = np.floor(np.random.uniform(0, 9.9, [BSIZE])).astype(np.int32)
        x = tf.constant(xval)
        y = tf.constant(yval)
        with tf.variable_scope("Model", reuse=None):
          m1 = get_multi_gpu_model(
              "resnet", config, num_replica=2, inp=x, label=y)
        with tf.variable_scope("Model", reuse=True):
          m = get_model("resnet", config, inp=x, label=y)
        with tf.variable_scope("Model", reuse=True):
          m2 = MultiPassModelV2(
              config,
              ResNetModel,
              num_passes=2,
              inp=x,
              label=y,
              aggregate_method=method)
        sess.run(tf.global_variables_initializer())
        m2.assign_lr(sess, config.base_learn_rate)
        tvars = tf.trainable_variables()
        tvars_str = map(lambda x: x.name, tvars)
        tvars_v0 = sess.run(tvars)
        tvars_d0 = dict(zip(tvars_str, tvars_v0))

        # Run multi tower version.
        saver = tf.train.Saver()
        saver.save(sess, CKPT_FNAME)
        for ii in range(3):
          m1.train_step(sess)
        tvars_v1 = sess.run(tvars)
        tvars_d1 = dict(zip(tvars_str, tvars_v1))

        # Run MultiPassModel.
        saver.restore(sess, CKPT_FNAME)
        for ii in range(3):
          m2.train_step(sess)
        tvars_v2 = sess.run(tvars)
        tvars_d2 = dict(zip(tvars_str, tvars_v2))

        for vv in tvars_str:
          np.testing.assert_allclose(
              tvars_d1[vv], tvars_d2[vv], rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
  tf.test.main()
