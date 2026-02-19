# Copyright 2026 DeepMind Technologies Limited
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
"""Visualize batch rendering output from MuJoCo Warp for debugging."""

import functools
import os
from typing import Sequence

from absl import app
from absl import flags
import jax
import jax.numpy as jp
import mediapy as media
import mujoco
from mujoco import mjx
from mujoco.mjx._src import bvh
from mujoco.mjx._src import forward
from mujoco.mjx._src import render
from mujoco.mjx._src import render_util
from mujoco.mjx._src import test_util
from mujoco.mjx.warp import io
from mujoco.mjx.warp import render as warp_render
from mujoco.mjx.warp import bvh as warp_bvh
from mujoco.mjx.warp.types import RenderContext
import numpy as np
import warp as wp


_MODELFILE = flags.DEFINE_string(
    'modelfile',
    'humanoid/humanoid.xml',
    'path to model',
)
_NWORLD = flags.DEFINE_integer('nworld', 4, 'number of worlds to render')
_WIDTH = flags.DEFINE_integer('width', 512, 'image width')
_HEIGHT = flags.DEFINE_integer('height', 512, 'image height')
_CAMERA_ID = flags.DEFINE_integer('camera_id', 0, 'camera id to visualize')
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', '/tmp/visualize_render', 'output directory'
)
_RANDOMIZE_QPOS = flags.DEFINE_boolean(
    'randomize_qpos', False, 'randomize initial qpos'
)
_USE_TEXTURES = flags.DEFINE_boolean('use_textures', True, 'enable textures')
_USE_SHADOWS = flags.DEFINE_boolean('use_shadows', True, 'enable shadows')
_USE_PMAP = flags.DEFINE_boolean('use_pmap', False, 'enable pmap')
_WP_KERNEL_CACHE_DIR = flags.DEFINE_string(
    'wp_kernel_cache_dir',
    '/tmp/wp_kernel_cache_dir_visualize_render',
    'warp kernel cache directory',
)

_COMPILER_OPTIONS = {'xla_gpu_graph_min_graph_size': 1}
jax_jit = functools.partial(jax.jit, compiler_options=_COMPILER_OPTIONS)


def _save_single(rgb, out_path):
  """Save first world as a single image."""
  img = np.asarray(rgb[0])
  img_uint8 = (img * 255).astype(np.uint8)
  media.write_image(out_path, img_uint8)
  print(f'  single image: {out_path}')


def _save_tiled(rgb, out_path):
  """Save all worlds as a tiled grid."""
  nworld, height, width, _ = rgb.shape
  cols = int(np.ceil(np.sqrt(nworld)))
  rows = int(np.ceil(nworld / cols))
  canvas = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)

  for w in range(nworld):
    img_uint8 = (np.asarray(rgb[w]) * 255).astype(np.uint8)
    r, c = w // cols, w % cols
    y0, y1 = r * height, (r + 1) * height
    x0, x1 = c * width, (c + 1) * width
    canvas[y0:y1, x0:x1, :] = img_uint8

  media.write_image(out_path, canvas)
  print(f'  tiled image:  {out_path}')


def _save_depth_single(depth, out_path):
  """Save first world depth as a single image."""
  # depth is (nworld, H, W) float in [0, 1]
  img = np.asarray(depth[0])
  img_uint8 = (img * 255).astype(np.uint8)
  # Expand to 3 channels for saving
  img_uint8 = np.stack([img_uint8]*3, axis=-1)
  media.write_image(out_path, img_uint8)
  print(f'  single depth: {out_path}')


def _save_depth_tiled(depth, out_path):
  """Save all worlds depth as a tiled grid."""
  nworld, height, width = depth.shape
  cols = int(np.ceil(np.sqrt(nworld)))
  rows = int(np.ceil(nworld / cols))
  canvas = np.zeros(
      (rows * height, cols * width), dtype=np.uint8
  )

  for w in range(nworld):
    img_uint8 = (np.asarray(depth[w]) * 255).astype(
        np.uint8
    )
    r, c = w // cols, w % cols
    y0, y1 = r * height, (r + 1) * height
    x0, x1 = c * width, (c + 1) * width
    canvas[y0:y1, x0:x1] = img_uint8

  # Expand to 3 channels
  canvas = np.stack([canvas]*3, axis=-1)
  media.write_image(out_path, canvas)
  print(f'  tiled depth:  {out_path}')


class DeviceDispatchDict(dict):
    """
    Dictionary that resolves device-specific context when accessing a key.

    If the value associated with a key is a dict of {device_alias: context},
    __getitem__ returns the context for the current Warp device.
    Otherwise, it returns the value directly.
    """

    def __getitem__(self, key):
      val = super().__getitem__(key)
      if isinstance(val, dict):
        device_alias = wp.get_device().alias
        if device_alias in val:
          return val[device_alias]
        # Fallback if alias not found (e.g., during tracing on CPU)
        # Return first available context for shape inference
        return next(iter(val.values()))
      return val


def render_pipeline(m, mx, worldids, rc):
  # Init
  @jax.vmap
  def init_fn(worldid):
    dx = mjx.make_data(m, impl='warp')
    rng = jax.random.PRNGKey(worldid)
    qpos0 = jp.array(m.qpos0)
    qpos = qpos0
    if _RANDOMIZE_QPOS.value:
      # TODO(robotics-team): consider integrating velocity if there are free
      # joints.
      qpos = qpos0 + jax.random.uniform(rng, (m.nq,), minval=-0.2, maxval=0.05)
    return dx.replace(qpos=qpos)

  dx_batch = jax.jit(init_fn)(worldids)

  # Forward
  dx_batch = jax.jit(jax.vmap(forward.forward, in_axes=(None, 0)))(mx, dx_batch)

  # Refit BVH
  dx_batch = jax.jit(jax.vmap(warp_bvh.refit_bvh, in_axes=(None, 0, None)))(
      mx, dx_batch, rc
  )

  # Forward
  # TODO(hartikainen): This forward is necessary. Otherwise the the renderer produces
  # weird-looking images with missing "slices" of geoms. I would `warp_bvh.refit_bvh` to
  # not touch `data` but for some reason it does.
  dx_batch = jax.jit(jax.vmap(forward.forward, in_axes=(None, 0)))(mx, dx_batch)
  
  # Render
  out_batch = jax.jit(jax.vmap(warp_render.render, in_axes=(None, 0, None)))(
      mx, dx_batch, rc
  )

  return out_batch


def _main(_: Sequence[str]):
  os.environ['MJX_WARP_ENABLED'] = 'true'

  wp.config.kernel_cache_dir = _WP_KERNEL_CACHE_DIR.value

  os.makedirs(_OUTPUT_DIR.value, exist_ok=True)

  try:
    m = test_util.load_test_file(_MODELFILE.value)
  except Exception:
    m = mujoco.MjModel.from_xml_path(_MODELFILE.value)

  print('visualize_render.py:\n')
  print(f'  modelfile   : {_MODELFILE.value}')
  print(f'  nworld      : {_NWORLD.value}')
  print(f'  resolution  : {_WIDTH.value}x{_HEIGHT.value}')
  print(f'  camera_id   : {_CAMERA_ID.value}')
  print(f'  use_textures: {_USE_TEXTURES.value}')
  print(f'  use_shadows : {_USE_SHADOWS.value}')
  print(f'  use_pmap    : {_USE_PMAP.value}')
  print(f'  output_dir  : {_OUTPUT_DIR.value}\n')

  mx = mjx.put_model(m, impl='warp')

  # Patch buffer dictionary
  dispatch_buffers = DeviceDispatchDict()
  io._MJX_RENDER_CONTEXT_BUFFERS = dispatch_buffers
  # Also patch render module's reference if it imported it separately
  warp_render._MJX_RENDER_CONTEXT_BUFFERS = dispatch_buffers
  warp_bvh._MJX_RENDER_CONTEXT_BUFFERS = dispatch_buffers

  # Monkey-patch shims to ensure correct Warp device scope in pmap threads.
  def patch_shim(original_shim):
      @functools.wraps(original_shim)
      def patched(*args):
          for arg in args:
              if hasattr(arg, 'device'):
                  with wp.ScopedDevice(arg.device):
                      return original_shim(*args)
          return original_shim(*args)
      return patched

  warp_render._render_shim = patch_shim(warp_render._render_shim)
  warp_bvh._refit_bvh_shim = patch_shim(warp_bvh._refit_bvh_shim)

  if _USE_PMAP.value:
    devices = jax.local_devices()
    num_devices = len(devices)
    print(f'Running with PMAP on {num_devices} devices: {devices}')

    nworld = _NWORLD.value
    if nworld % num_devices != 0:
      raise ValueError(
          f'nworld ({nworld}) must be divisible by num_devices ({num_devices})'
      )
    nworld_per_device = nworld // num_devices

    # Create contexts for each device
    multi_device_ctx = {}
    ctx_key = 99999

    for i, device in enumerate(devices):
      wp_device = f'cuda:{device.id}'
      with wp.ScopedDevice(wp_device):
        print(f'Creating context for device {wp_device}...')
        rc_wrapper_returned = io.create_render_context(
            mjm=m,
            nworld=nworld_per_device,
            cam_res=(_WIDTH.value, _HEIGHT.value),
            use_textures=_USE_TEXTURES.value,
            use_shadows=_USE_SHADOWS.value,
            render_rgb=True,
            render_depth=True,
            enabled_geom_groups=[0, 1, 2],
        )
        rc = dispatch_buffers.pop(rc_wrapper_returned.key)
        multi_device_ctx[wp_device] = rc

    dispatch_buffers[ctx_key] = multi_device_ctx
    rc_wrapper = RenderContext(ctx_key, _owner=False)

    worldids = jp.arange(nworld).reshape(num_devices, nworld_per_device)
    mx_replicated = jax.device_put_replicated(mx, devices)
    
    pmap_pipeline = jax.pmap(functools.partial(render_pipeline, m), axis_name='device')

    print('running pmap pipeline...')
    out_pmap = pmap_pipeline(mx_replicated, worldids, rc_wrapper)
    out_pmap = jax.block_until_ready(out_pmap)

    rgb_pmap = out_pmap[0]
    depth_pmap = out_pmap[1]
    
    rgb_packed = rgb_pmap.reshape((nworld,) + rgb_pmap.shape[2:])
    depth_packed = depth_pmap.reshape((nworld,) + depth_pmap.shape[2:])
    
    rgb = jax.vmap(render_util.get_rgb, in_axes=(None, None, 0))(
      rc_wrapper, _CAMERA_ID.value, rgb_packed
    )
    depth = jax.vmap(render_util.get_depth, in_axes=(None, None, 0, None))(
      rc_wrapper, _CAMERA_ID.value, depth_packed, 4.0
    )

    single_path = os.path.join(_OUTPUT_DIR.value, f'camera_{_CAMERA_ID.value}.png')
    _save_single(rgb, single_path)
    depth_single_path = os.path.join(_OUTPUT_DIR.value, f'depth_{_CAMERA_ID.value}.png')
    _save_depth_single(depth, depth_single_path)

    if nworld > 1:
      tiled_path = os.path.join(_OUTPUT_DIR.value, f'tiled_{_CAMERA_ID.value}.png')
      _save_tiled(rgb, tiled_path)
      depth_tiled_path = os.path.join(_OUTPUT_DIR.value, f'depth_tiled_{_CAMERA_ID.value}.png')
      _save_depth_tiled(depth, depth_tiled_path)

    print('\ndone (pmap).')
    return

  worldids = jp.arange(_NWORLD.value)

  print('creating render context...')
  rc = io.create_render_context(
      mjm=m,
      nworld=_NWORLD.value,
      cam_res=(_WIDTH.value, _HEIGHT.value),
      use_textures=_USE_TEXTURES.value,
      use_shadows=_USE_SHADOWS.value,
      render_rgb=True,
      render_depth=True,
      enabled_geom_groups=[0, 1, 2],
  )

  print('running pipeline...')
  out_batch = render_pipeline(m, mx, worldids, rc)
  out_batch = jax.block_until_ready(out_batch)

  rgb_packed = out_batch[0]
  depth_packed = out_batch[1]
  print(f'  rgb shape:   {rgb_packed.shape}')
  print(f'  depth shape: {depth_packed.shape}\n')

  rgb = jax.vmap(render_util.get_rgb, in_axes=(None, None, 0))(
      rc, _CAMERA_ID.value, rgb_packed
  )

  depth = jax.vmap(render_util.get_depth, in_axes=(None, None, 0, None))(
      rc, _CAMERA_ID.value, depth_packed, 10.0
  )

  single_path = os.path.join(
      _OUTPUT_DIR.value, f'camera_{_CAMERA_ID.value}.png'
  )

  single_path = os.path.join(_OUTPUT_DIR.value, f'camera_{_CAMERA_ID.value}.png')
  _save_single(rgb, single_path)
  depth_single_path = os.path.join(_OUTPUT_DIR.value, f'depth_{_CAMERA_ID.value}.png')
  _save_depth_single(depth, depth_single_path)

  depth_rgb = np.repeat(np.asarray(depth)[..., None], 3, axis=-1)
  depth_single_path = os.path.join(
      _OUTPUT_DIR.value, f'depth_{_CAMERA_ID.value}.png'
  )
  _save_single(depth_rgb, depth_single_path)

  if _NWORLD.value > 1:
    tiled_path = os.path.join(_OUTPUT_DIR.value, f'tiled_{_CAMERA_ID.value}.png')
    _save_tiled(rgb, tiled_path)
    depth_tiled_path = os.path.join(_OUTPUT_DIR.value, f'depth_tiled_{_CAMERA_ID.value}.png')
    _save_depth_tiled(depth, depth_tiled_path)

    depth_tiled_path = os.path.join(
        _OUTPUT_DIR.value, f'depth_tiled_{_CAMERA_ID.value}.png'
    )
    _save_tiled(depth_rgb, depth_tiled_path)

  print('\ndone.')


def main():
  app.run(_main)


if __name__ == '__main__':
  main()
