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
from mujoco.mjx.warp import ffi
from mujoco.mjx.warp import io
from mujoco.mjx.warp.io import _MJX_RENDER_CONTEXT_BUFFERS
from mujoco.mjx.warp import render as warp_render
from mujoco.mjx.warp import bvh as warp_bvh
from mujoco.mjx.warp.types import RenderContext
import numpy as np
import sys
import warp as wp


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


_ORIGINAL_BUFFERS = {}

def setup_multigpu_contexts(m, nworld_per_device, devices, width, height, use_textures, use_shadows):
  """Sets up the multi-GPU rendering environment by patching global buffers."""
  virtual_key = 999999
  
  # Create the smart dictionary
  smart_dict = DeviceDispatchDict()
  # Patch modules FIRST so that io.create_render_context writes to our smart_dict
  _ORIGINAL_BUFFERS['warp_render'] = warp_render._MJX_RENDER_CONTEXT_BUFFERS
  _ORIGINAL_BUFFERS['warp_bvh'] = warp_bvh._MJX_RENDER_CONTEXT_BUFFERS
  _ORIGINAL_BUFFERS['io'] = io._MJX_RENDER_CONTEXT_BUFFERS

  warp_render._MJX_RENDER_CONTEXT_BUFFERS = smart_dict
  warp_bvh._MJX_RENDER_CONTEXT_BUFFERS = smart_dict
  io._MJX_RENDER_CONTEXT_BUFFERS = smart_dict

  rcs = []
  virtual_key_map = {}

  # Create contexts for each device
  for d in devices:
    with wp.ScopedDevice(f'cuda:{d.id}'):
      rc = io.create_render_context(
          mjm=m,
          nworld=nworld_per_device,
          cam_res=(width, height),
          use_textures=use_textures,
          use_shadows=use_shadows,
          render_rgb=True,
          render_depth=True,
          enabled_geom_groups=[0, 1, 2],
      )
      rcs.append(rc)
      
      # Retrieve the underlying Warp object
      real_warp_rc = smart_dict[rc.key]
      
      # Map device alias to real object
      virtual_key_map[f'cuda:{d.id}'] = real_warp_rc

  smart_dict[virtual_key] = virtual_key_map

  # Return a virtual RC handle and the list of real RCs (for post-processing)
  virtual_rc = RenderContext(key=virtual_key, _owner=False)
  return virtual_rc, rcs


def restore_multigpu_contexts():
  """Restores the original global buffers."""
  if not _ORIGINAL_BUFFERS:
    return
  warp_render._MJX_RENDER_CONTEXT_BUFFERS = _ORIGINAL_BUFFERS['warp_render']
  warp_bvh._MJX_RENDER_CONTEXT_BUFFERS = _ORIGINAL_BUFFERS['warp_bvh']
  io._MJX_RENDER_CONTEXT_BUFFERS = _ORIGINAL_BUFFERS['io']
  _ORIGINAL_BUFFERS.clear()


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

    # Set up multi-GPU environment
    virtual_rc, rcs = setup_multigpu_contexts(
        m, nworld_per_device, devices, 
        _WIDTH.value, _HEIGHT.value, 
        _USE_TEXTURES.value, _USE_SHADOWS.value
    )

    worldids = jp.arange(nworld).reshape(num_devices, nworld_per_device)
    
    # Replicate mx to all devices
    mx = jax.device_put_replicated(mx, devices)

    pmap_pipeline = jax.pmap(
        render_pipeline,
        in_axes=(None, 0, 0, None),
        static_broadcasted_argnums=(0, 3), # m and rc are static
    )

    print('running pmap pipeline...')
    out_pmap = pmap_pipeline(m, mx, worldids, virtual_rc)
    out_pmap = jax.block_until_ready(out_pmap)

    rgb_pmap = out_pmap[0]
    depth_pmap = out_pmap[1]
    
    # Combine results from devices
    rgb_packed = rgb_pmap.reshape((nworld,) + rgb_pmap.shape[2:])
    depth_packed = depth_pmap.reshape((nworld,) + depth_pmap.shape[2:])
    
    rgb = jax.vmap(render_util.get_rgb, in_axes=(None, None, 0))(
      rc, _CAMERA_ID.value, rgb_packed
    )
    depth = jax.vmap(render_util.get_depth, in_axes=(None, None, 0, None))(
      rc, _CAMERA_ID.value, depth_packed, 4.0
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
