"""Tests for render functions."""

import datetime
import os
from pathlib import Path
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
import jax
from jax import numpy as jp
import mediapy as media
import mujoco
from mujoco import mjx
from mujoco.mjx.warp import render as mjxw_render
from mujoco.mjx._src import io as mjx_io
from mujoco.mjx._src import render
import mujoco.mjx.warp as mjxw
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import warp as wp


_MULTIPLE_CONVEX_OBJECTS = (epath.Path(__file__).parent / 'render_model.xml').read_text()


def extract_and_resize(pixels, resolutions, mask, max_res, is_rgb):
  outputs = []
  cur = 0
  for i, active in enumerate(mask):
    if active:
      w, h = int(resolutions[i][0]), int(resolutions[i][1])
      # chunk = jax.lax.slice(pixels, (cur,), (cur + w * h,))
      chunk = jax.lax.slice_in_dim(pixels, cur, cur + w * h, axis=-1)
      img = chunk.reshape(*chunk.shape[:-1], h, w)
      if is_rgb:
        r = ((img >> 16) & 0xFF).astype(jp.uint8)
        g = ((img >> 8) & 0xFF).astype(jp.uint8)
        b = ((img >> 0) & 0xFF).astype(jp.uint8)
        img = jp.stack([r, g, b], axis=-1)
        img = jax.image.resize(img, (max_res[1], max_res[0], 3), method='nearest')
        img = img.astype(jp.uint8)
      else:
        img = jax.image.resize(img, (max_res[1], max_res[0]), method='nearest')
      outputs.append(img)
      cur += w * h

  return jp.stack(outputs)


def save_video(
    frames_rgb,
    frames_depth,
    m,
    width,
    height,
    path: Path | str = Path('/tmp/warp_bvh'),
    prefix='video',
    fps=60,
    num_cameras=None,
    layout=None,
):
  """Saves video from frames."""
  path = Path(path)
  path.mkdir(parents=True, exist_ok=True)
  timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

  # Ensure numpy
  frames_rgb = np.array(frames_rgb)
  frames_depth = np.array(frames_depth)

  # Shapes
  # Expecting: (T, [Devices], Batch, [1], Cam*H*W) or similar variants
  # We want to normalize to: (T, D, B, C, H, W)

  # 1. Identify T, D, B
  if frames_rgb.ndim == 7:  # (T, D, B, C, H, W, 3)
    T, D, B, C, H, W, _ = frames_rgb.shape
    frames_rgb_unpacked = frames_rgb
  elif frames_rgb.ndim == 6:  # (T, D, B, C, H, W)
    T, D, B, C, H, W = frames_rgb.shape
    # Unpack RGB (uint32 -> uint8 channels)
    # NOTE(hartikainen): `pack_rgba_to_uint32` in `render.py` packs as:
    # (A << 24) | (R << 16) | (G << 8) | B
    r = ((frames_rgb >> 16) & 0xFF).astype(np.uint8)
    g = ((frames_rgb >> 8) & 0xFF).astype(np.uint8)
    b = ((frames_rgb >> 0) & 0xFF).astype(np.uint8)
    frames_rgb_unpacked = np.stack([r, g, b], axis=-1)
  else:
    if frames_rgb.ndim == 5:  # (T, D, B, ?, ?)
      T, D, B = frames_rgb.shape[:3]
    elif frames_rgb.ndim == 4:  # (T, B, ?, ?) - Single device case (vmap) treated as D=1
      T, B = frames_rgb.shape[:2]
      D = 1
    else:
      raise ValueError(f'Unexpected shape: {frames_rgb.shape}')

    C = num_cameras or m.ncam

    # 2. Reshape to target
    # This flattens the tail dimensions and reconstructs C, H, W
    # This handles cases where C is its own dim or merged with H*W
    frames_rgb = frames_rgb.reshape(T, D, B, C, height, width)

    # Unpack RGB (uint32 -> uint8 channels)
    # NOTE(hartikainen): `pack_rgba_to_uint32` in `render.py` packs as:
    # (A << 24) | (R << 16) | (G << 8) | B
    r = ((frames_rgb >> 16) & 0xFF).astype(np.uint8)
    g = ((frames_rgb >> 8) & 0xFF).astype(np.uint8)
    b = ((frames_rgb >> 0) & 0xFF).astype(np.uint8)
    frames_rgb_unpacked = np.stack([r, g, b], axis=-1)

  # Normalize Depth
  frames_depth = frames_depth.reshape(T, D, B, C, height, width)
  d_min = np.percentile(frames_depth, 0)
  d_max = np.percentile(frames_depth, 90)
  frames_depth_norm = np.clip((frames_depth - d_min) / (d_max - d_min + 1e-6), 0, 1)
  frames_depth_uint8 = (frames_depth_norm * 255).astype(np.uint8)
  frames_depth_unpacked = np.stack([frames_depth_uint8] * 3, axis=-1)

  if layout == 'vertical_cameras':
    rows = C
    cols = D * B
  else:
    rows = D  # Rows = Devices
    cols = B * C  # Cols = Batch * Cam

  camera_names = [m.camera(i).name for i in range(m.ncam)]

  def make_grid(frames_data):
    video = []

    try:
      font = ImageFont.truetype("LiberationMono-Regular", 24)
    except IOError:
      font = ImageFont.load_default()

    for t in range(T):
      full_width = cols * width
      full_height = rows * height
      img = Image.new('RGB', (full_width, full_height))
      draw = ImageDraw.Draw(img)

      for d in range(D):
        for b in range(B):
          for c in range(C):
            tile = frames_data[t, d, b, c]
            pil_tile = Image.fromarray(tile)

            # Grid position
            if layout == 'vertical_cameras':
              row_idx = c
              col_idx = d * B + b
            else:
              # Row depends on Device (d)
              # Col depends on Batch (b) and Camera (c)
              row_idx = d
              col_idx = b * C + c

            x = col_idx * width
            y = row_idx * height
            img.paste(pil_tile, (x, y))

            # Label
            label = f'Cam: {camera_names[c]} | Dev {d} World {b}'

            # Draw text with shadow for visibility
            text_pos = (x + 10, y + 10)
            bbox = draw.textbbox(text_pos, label, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0, 128)) # Semi-transparent background if possible, but PIL RGB doesn't support alpha on draw directly easily without RGBA.
            # Just shadow
            draw.text((text_pos[0]+2, text_pos[1]+2), label, font=font, fill=(0, 0, 0))
            draw.text(text_pos, label, font=font, fill=(255, 255, 255))

      video.append(np.array(img))
    return video

  video_rgb = make_grid(frames_rgb_unpacked)
  video_depth = make_grid(frames_depth_unpacked)

  media.write_video(
      path / f'{prefix}_{timestamp}_rgb.mp4', video_rgb, fps=fps
  )
  media.write_video(
      path / f'{prefix}_{timestamp}_depth.mp4', video_depth, fps=fps
  )


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

    spec = mujoco.MjSpec.from_string(_MULTIPLE_CONVEX_OBJECTS)
    m = spec.compile()
    mx = mjx.put_model(m, impl='warp')
    nworld = 4
    width, height = 640, 480
    n_devices = jax.local_device_count()

    active_camera_ids = mx.bind([spec.camera("camera-1"), spec.camera("camera-2")]).id
    # Different resolutions for each camera
    camera_resolutions = (
      np.array([width, height])[None, :] / (2 ** np.arange(mx.ncam))[:, None]
    ).astype(int)
    camera_mask = np.isin(np.arange(mx.ncam), active_camera_ids)
    n_active_cameras = len(active_camera_ids)

    def reset(key) -> tuple[mjx.Data, tuple[jp.ndarray, jp.ndarray]]:
      key = jax.random.split(key, nworld)
      dx = jax.vmap(lambda key: mjx.make_data(m, impl='warp'))(key)
      qpos = jax.vmap(
        lambda qpos, key: (
          qpos + jax.random.uniform(key, (m.nq,), minval=-1.0, maxval=+1.0)
        )
      )(dx.qpos, key)
      dx = dx.replace(qpos=qpos)

      with mjxw_render._RENDER_CONTEXT_LOCK:
        rc_id = len(mjxw_render._RENDER_CONTEXT_BUFFERS)

      # TODO(hartikainen): We need to create render context for each device here because
      # the render context is put on a device at the creation time. We don't know what
      # device we're on when the `jax.pmap` function is executed, so we need to create
      # it on all devices here, and then select the correct one at runtime **inside** the
      # jax-warp ffi call.
      for device in map(wp.device_from_jax, jax.devices()):
        with wp.ScopedDevice(device):
          unused_rc = render.create_render_context(
              m,
              mx,
              dx,
              cam_res=camera_resolutions[camera_mask],
              render_rgb=True,
              render_depth=True,
              use_textures=False,
              use_shadows=False,
              enabled_geom_groups=[0, 1, 2],
              cam_active=camera_mask,
              key=(rc_id, str(device)),
          )

      # NOTE(hartikainen): We don't use device-specific render context because we
      # don't know what device the render will be called from. Thus, we store just
      # the integer key and create the render context on-demand inside jax-warp ffi
      # call.

      rc_id_without_device = mjxw_render.RenderContextRegistry(key=rc_id)
      # NOTE(hartikainen): What `_device_specific_rc_id` does is that it tells the
      # render function to look up the correct render context based on the
      # runtime device the jax function is being executed on. This is needed because
      # we don't know here what device we'll be rendering on.
      dx = dx.replace(_render_context=rc_id_without_device, _device_specific_rc_id=True)

      rgb, depth = render.render(mx, dx, dx._render_context)
      rgb = jax.vmap(
        lambda x: extract_and_resize(x, camera_resolutions, camera_mask, (width, height), True)
      )(rgb)
      depth = jax.vmap(
        lambda x: extract_and_resize(x, camera_resolutions, camera_mask, (width, height), False)
      )(depth)

      return dx, (rgb, depth)

    def step(dx) -> tuple[mjx.Data, tuple[jp.ndarray, jp.ndarray]]:
      def body(i, dx):
        return mjx.step(mx, dx)
      dx = jax.lax.fori_loop(0, steps_per_render, body, dx)

      rgb, depth = render.render(mx, dx, dx._render_context)
      rgb = jax.vmap(
        lambda x: extract_and_resize(x, camera_resolutions, camera_mask, (width, height), True)
      )(rgb)
      depth = jax.vmap(
        lambda x: extract_and_resize(x, camera_resolutions, camera_mask, (width, height), False)
      )(depth)

      return dx, (rgb, depth)

    def verify_output(rgb: jp.ndarray, depth: jp.ndarray):
      # TODO(hartikainen): Something's wrong with the shape. Why is there an extra unit
      # dimension?
      self.assertEqual(rgb.shape, (n_devices, nworld, n_active_cameras, height, width, 3))
      self.assertEqual(rgb.dtype, jp.uint8)
      self.assertEqual(depth.shape, (n_devices, nworld, n_active_cameras, height, width))
      self.assertEqual(depth.dtype, jp.float32)

    fps = 60
    render_dt = 1 / fps
    sim_dt = m.opt.timestep
    steps_per_render = int(render_dt / sim_dt)

    step = jax.pmap(step)
    reset = jax.pmap(reset)

    key = jax.random.split(jax.random.PRNGKey(0), n_devices)

    dx, (rgb, depth) = reset(key)
    verify_output(rgb, depth)
    frames_rgb = [rgb]
    frames_depth = [depth]

    for i in range(60):
      dx, (rgb, depth) = step(dx)
      verify_output(rgb, depth)
      frames_rgb.append(rgb)
      frames_depth.append(depth)

    save_video(
      frames_rgb,
      frames_depth,
      m,
      width,
      height,
      prefix='test_pmap_render_context',
      fps=fps,
      num_cameras=n_active_cameras,
    )

  @mock.patch.dict(os.environ, {'MJX_GPU_DEFAULT_WARP': 'true'})
  def test_render_context(self):
    """Tests that RenderContext works under jax.vmap."""
    if not mjxw.WARP_INSTALLED:
      self.skipTest('Warp not installed.')
    if not mjx_io.has_cuda_gpu_device():
      self.skipTest('No CUDA GPU device available.')

    spec = mujoco.MjSpec.from_string(_MULTIPLE_CONVEX_OBJECTS)
    m = spec.compile()
    mx = mjx.put_model(m, impl='warp')
    nworld = 4
    width, height = 320, 240

    dummy_dx = jax.vmap(lambda _: mjx.make_data(m, impl='warp'))(jp.arange(nworld))
    rc = render.create_render_context(
      m,
      mx,
      dummy_dx,
      cam_res=(width, height),
      render_rgb=True,
      render_depth=True,
      use_textures=False,
      use_shadows=False,
      enabled_geom_groups=[0, 1, 2],
      cam_active=[True] * m.ncam,
    )

    @jax.vmap
    def reset_and_render(key):
      dx = mjx.make_data(m, impl='warp')
      qpos = dx.qpos + jax.random.uniform(key, (m.nq,), minval=-0.1, maxval=+0.1)
      dx = dx.replace(qpos=qpos)
      pixels = render.render(mx, dx, rc)
      return pixels

    key = jax.random.split(jax.random.PRNGKey(0), nworld)
    pixels_rgb, pixels_depth = reset_and_render(key)

    self.assertEqual(pixels_rgb.shape, (nworld, 1, m.ncam * width * height))
    self.assertEqual(pixels_rgb.dtype, jp.uint32)
    self.assertEqual(pixels_depth.shape, (nworld, 1, m.ncam * width * height))
    self.assertEqual(pixels_depth.dtype, jp.float32)


if __name__ == '__main__':
  absltest.main()
