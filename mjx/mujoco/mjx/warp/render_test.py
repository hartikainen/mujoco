# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for codegen'd smooth functions."""

import dataclasses
import functools
import os
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import io
from mujoco.mjx._src import render
from mujoco.mjx._src import forward
import mujoco.mjx.warp as mjxw
from mujoco.mjx.warp import test_util as tu
from mujoco.mjx.warp import warp as wp  # pylint: disable=g-importing-member
import numpy as np
import mediapy as media
from mujoco.mjx.third_party import mujoco_warp as mjw

_FORCE_TEST = os.environ.get('MJX_WARP_FORCE_TEST', '0') == '1'


def save_image(pixels, cam_id, width, height, out_fpath='test.png'):
  pixels = pixels[cam_id]
  pixels = pixels.reshape((width, height))
  r = (pixels & 0xFF).astype(np.uint8)
  g = ((pixels >> 8) & 0xFF).astype(np.uint8)
  b = ((pixels >> 16) & 0xFF).astype(np.uint8)
  pixels = np.dstack([r, g, b])
  media.write_image(out_fpath, pixels)


class RenderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if mjxw.WARP_INSTALLED:
      # self.tempdir = tempfile.TemporaryDirectory()
      tempdir = '/tmp/wp_kernel_cache_dir_RenderTest'
      # wp.config.kernel_cache_dir = self.tempdir.name
      wp.config.kernel_cache_dir = tempdir
    np.random.seed(0)

  def tearDown(self):
    super().tearDown()
    if hasattr(self, 'tempdir'):
      self.tempdir.cleanup()

  def test_render(self):
    """Tests render."""
    if not _FORCE_TEST:
      if not mjxw.WARP_INSTALLED:
        self.skipTest('Warp not installed.')
      if not io.has_cuda_gpu_device():
        self.skipTest('No CUDA GPU device available.')

    m = tu.load_test_file('humanoid/humanoid.xml')
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    camera_id = 1
    width, height = 512, 512

    # MJX model/data configured for Warp
    mx = mjx.put_model(m, impl='warp')
    dx = mjx.make_data(m, impl='warp')
    dx = dx.replace(qpos=d.qpos)

    # Attach registry id to the MJX model
    rx = mjx.create_render_context(
      m,
      mx,
      dx,
      nworld=1,
      width=width,
      height=height,
      use_textures=True,
      use_shadows=True,
      fov_rad=wp.radians(60.0),
      render_rgb=True,
      render_depth=True,
      enabled_geom_groups=[0, 1, 2],
    )

    # Forward kinematics and render via JAX → outputs written into rc buffers
    dx = jax.jit(forward.forward)(mx, dx)
    _ = jax.jit(render.render)(mx, dx, rx)
    save_image(rx.pixels.numpy()[0], camera_id, width, height)


if __name__ == '__main__':
  absltest.main()