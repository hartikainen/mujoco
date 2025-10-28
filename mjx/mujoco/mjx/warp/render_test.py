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


def save_image(pixels, nworld, cam_id, width, height, out_fpath='test.png'):

  # Ensure numpy array on host
  arr = np.asarray(pixels)

  # Unpack packed uint32 RGB into uint8 HxW x 3
  def unpack_to_rgb(img_flat):
    img = img_flat.reshape((height, width))
    r = (img & 0xFF).astype(np.uint8)
    g = ((img >> 8) & 0xFF).astype(np.uint8)
    b = ((img >> 16) & 0xFF).astype(np.uint8)
    return np.dstack([r, g, b])

  if nworld == 1:
    # Expect arr shape: (ncam, H*W) or (1, ncam, H*W)
    if arr.ndim == 3:
      img_flat = arr[0, cam_id]
    else:
      img_flat = arr[cam_id]
    rgb = unpack_to_rgb(img_flat)
    media.write_image(out_fpath, rgb)
    return

  cols = int(np.ceil(np.sqrt(nworld)))
  rows = int(np.ceil(nworld / cols))

  canvas = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
  for w in range(nworld):
    img_flat = arr[w, cam_id]
    rgb = unpack_to_rgb(img_flat)
    r = w // cols
    c = w % cols
    y0, y1 = r * height, (r + 1) * height
    x0, x1 = c * width, (c + 1) * width
    canvas[y0:y1, x0:x1, :] = rgb

  media.write_image(out_fpath, canvas)

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
  
  @parameterized.product(
      xml=(
          'humanoid/humanoid.xml',
      ),
      batch_size=(1, 16),
  )
  def test_render(self, xml: str, batch_size: int):
    """Tests render."""
    if not _FORCE_TEST:
      if not mjxw.WARP_INSTALLED:
        self.skipTest('Warp not installed.')
      if not io.has_cuda_gpu_device():
        self.skipTest('No CUDA GPU device available.')

    m = tu.load_test_file(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    mx = mjx.put_model(m, impl='warp')

    worldids = jp.arange(batch_size)
    dx_batch = jax.vmap(functools.partial(tu.make_data, m))(worldids)

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, batch_size)
    qpos0 = jp.array(m.qpos0)
    rand_qpos = jax.vmap(lambda k: qpos0 + jax.random.uniform(k, (m.nq,), minval=-0.2, maxval=0.05))(keys)
    dx_batch = jax.vmap(lambda dx, q: dx.replace(qpos=q))(dx_batch, rand_qpos)

    dx_batch = jax.jit(jax.vmap(forward.forward, in_axes=(None, 0)))(
        mx, dx_batch
    )
    

    print(dx_batch.qpos)

    camera_id = 1
    width, height = 512, 512
    render_context = mjx.create_render_context(
      m,
      mx,
      dx_batch,
      nworld=batch_size,
      width=width,
      height=height,
      use_textures=True,
      use_shadows=True,
      fov_rad=wp.radians(60.0),
      render_rgb=True,
      render_depth=True,
      enabled_geom_groups=[0, 1, 2],
    )

    out_batch = jax.jit(jax.vmap(render.render, in_axes=(None, 0, None)), static_argnums=(2,))(
        mx, dx_batch, render_context
    )
    # Save a single image: either the sole world, or a tiled grid for multi-world
    rgb_all_worlds = out_batch[0]  # shape: (nworld, ncam, H*W)
    out_path = 'tiled.png' if batch_size > 1 else '0.png'
    save_image(rgb_all_worlds, batch_size, camera_id, width, height, out_path)

  
  def test_warp_render(self):
    """Tests warp render."""
    m = tu.load_test_file('humanoid/humanoid.xml')
    camera_id = 1
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    mw = mjw.put_model(m)
    dw = mjw.put_data(m, d, nworld=1)
    dw.qpos = wp.array([d.qpos], dtype=wp.float32)
    mjw.forward(mw, dw)

    rc = mjw.create_render_context(
      m,
      mw,
      dw,
      nworld=1,
      width=512,
      height=512,
      use_textures=True,
      use_shadows=True,
      fov_rad=wp.radians(60.0),
      render_rgb=True,
      render_depth=True,
      enabled_geom_groups=[0, 1, 2],
    )

    mjw.render(mw, dw, rc)

    save_image(rc.pixels.numpy(), 1, camera_id, 512, 512, 'test_warp_render.png')


if __name__ == '__main__':
  absltest.main()