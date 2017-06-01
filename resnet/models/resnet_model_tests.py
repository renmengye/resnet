from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from resnet.configs import test_configs
from resnet.configs import get_config
from resnet.models.resnet_model import ResNetModel
from resnet.models.model_factory import get_model
from resnet.utils import logger

log = logger.get()


class ResNetModelTests(tf.test.TestCase):
  """Tests the single for-loop implementation is the same as the double
    for-loop implementation."""

  def _test_getmodel(self, modelname):
    with tf.Graph().as_default(), self.test_session(
    ) as sess, log.verbose_level(2):
      config = get_config("test", modelname)
      np.random.seed(0)
      xval = np.random.uniform(0.0, 1.0, [
          config.batch_size, config.height, config.width, config.num_channel
      ]).astype(np.float32)
      x = tf.constant(xval)
      with tf.variable_scope("Model", reuse=None):
        m = get_model("resnet", config, is_training=True, inp=x)

      sess.run(tf.global_variables_initializer())
      y = m.infer_step(sess)

  def test_getmodel(self):
    """Tests initialize ResNet object."""
    self._test_getmodel("resnet-test")

  def test_getbtlmodel(self):
    """Tests initialize ResNet object with bottleneck."""
    self._test_getmodel("resnet-btl-test")


if __name__ == "__main__":
  tf.test.main()
