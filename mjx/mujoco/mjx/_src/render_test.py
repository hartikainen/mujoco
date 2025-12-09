"""Tests for render functions."""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx.warp import render as mjxw_render
from mujoco.mjx._src import io as mjx_io
from mujoco.mjx._src import render
import mujoco.mjx.warp as mjxw
import numpy as np
import warp as wp


_MULTIPLE_CONVEX_OBJECTS = (epath.Path(__file__).parent / 'render_model.xml').read_text()


class RenderTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if mjxw.WARP_INSTALLED:
      self.tempdir = tempfile.TemporaryDirectory()
      # wp.config.kernel_cache_dir = self.tempdir.name

  @mock.patch.dict(os.environ, {'MJX_GPU_DEFAULT_WARP': 'true'})
  def test_pmap_render_context(self):
    """Tests that RenderContext works under jax.pmap."""
    if not mjxw.WARP_INSTALLED:
      self.skipTest('Warp not installed.')
    if not mjx_io.has_cuda_gpu_device():
      self.skipTest('No CUDA GPU device available.')

    m = mujoco.MjModel.from_xml_string(_MULTIPLE_CONVEX_OBJECTS)
    mx = mjx.put_model(m, impl='warp')
    nworld = 4
    width, height = 1920, 1280
    n_devices = jax.local_device_count()

    @jax.pmap
    def reset_and_render(key):
      key = jax.random.split(key, nworld)
      dx = jax.vmap(lambda key: mjx.make_data(m, impl='warp'))(key)
      qpos = jax.vmap(
        lambda qpos, key: (
          qpos + jax.random.uniform(key, (m.nq,), minval=-0.1, maxval=+0.1)
        )
      )(dx.qpos, key)
      dx = dx.replace(qpos=qpos)

      with mjxw_render._RENDER_CONTEXT_LOCK:
        rc_id = len(mjxw_render._RENDER_CONTEXT_BUFFERS)

      for device in map(wp.device_from_jax, jax.devices()):
        with wp.ScopedDevice(device):
          unused_rc = render.create_render_context(
              m, mx, dx, nworld=qpos.shape[0], width=width, height=height,
              use_textures=False, use_shadows=False, fov_rad=0.7,
              render_rgb=True, render_depth=True, key=(rc_id, str(device))
          )

      # NOTE(hartikainen): We don't use device-specific render context because we
      # don't know what device the render will be called from. Thus, we store just
      # the integer key and create the render context on-demand inside jax-warp ffi
      # call.

      rc_id_without_device = mjxw_render.RenderContextRegistry(key=rc_id)
      dx = dx.replace(
          _render_context=rc_id_without_device,
          # NOTE(hartikainen): What `_device_specific_rc_id` does is that it tells the
          # render function to look up the correct render context based on the
          # runtime device the jax function is being executed on. This is needed because
          # we don't know here what device we'll be rendering on.
          _device_specific_rc_id=True,
      )

      pixels = render.render(mx, dx, rc_id_without_device)
      return pixels

    key = jax.random.split(jax.random.PRNGKey(0), n_devices)
    pixels_depth, pixels_rgb = reset_and_render(key)

    self.assertEqual(pixels_depth.shape, (n_devices, nworld, m.ncam, width * height))
    self.assertEqual(pixels_rgb.shape, (n_devices, nworld, m.ncam, width * height))

  @mock.patch.dict(os.environ, {'MJX_GPU_DEFAULT_WARP': 'true'})
  def test_render_context(self):
    """Tests that RenderContext works under jax.vmap."""
    if not mjxw.WARP_INSTALLED:
      self.skipTest('Warp not installed.')
    if not mjx_io.has_cuda_gpu_device():
      self.skipTest('No CUDA GPU device available.')

    m = mujoco.MjModel.from_xml_string(_MULTIPLE_CONVEX_OBJECTS)
    mx = mjx.put_model(m, impl='warp')
    width, height = 1920, 1280
    nworld = 4

    dummy_dx = jax.vmap(lambda _: mjx.make_data(m, impl='warp'))(jp.arange(nworld))
    rc = render.create_render_context(
        m, mx, dummy_dx, nworld=nworld, width=width, height=height,
        use_textures=False, use_shadows=False, fov_rad=0.7,
        render_rgb=True, render_depth=True
    )

    @jax.vmap
    def reset_and_render(key):
      dx = mjx.make_data(m, impl='warp')
      qpos = dx.qpos + jax.random.uniform(key, (m.nq,), minval=-0.1, maxval=+0.1)
      dx = dx.replace(qpos=qpos)
      pixels = render.render(mx, dx, rc)
      return pixels

    key = jax.random.split(jax.random.PRNGKey(0), nworld)
    pixels_depth, pixels_rgb = reset_and_render(key)

    self.assertEqual(pixels_depth.shape, (nworld, m.ncam, width * height))
    self.assertEqual(pixels_rgb.shape, (nworld, m.ncam, width * height))


if __name__ == '__main__':
  absltest.main()
