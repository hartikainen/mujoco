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
import concurrent.futures
import gc
import tempfile
import weakref

from absl.testing import absltest
import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco.mjx._src import types
import mujoco.mjx.warp as mjxw
from mujoco.mjx.warp import warp as wp
import numpy as np


_XML = """
<mujoco>
  <visual><headlight ambient="0.3 0.3 0.3"/></visual>
  <worldbody>
    <light pos="0 0 3" diffuse="1 1 1"/>
    <camera pos="0 0 3" xyaxes="1 0 0 0 1 0" fovy="45"/>
    <geom type="plane" size="2 2 0.1" pos="0 0 -0.5"/>
    <body>
      <freejoint/>
      <geom type="sphere" size="0.4" rgba="0.9 0.2 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""

_FLEX_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" diffuse="1 1 1"/>
    <camera pos="0 0 3" xyaxes="1 0 0 0 1 0" fovy="45"/>
    <geom type="plane" size="2 2 0.1" pos="0 0 -0.5"/>
    <flexcomp name="cloth" type="grid" dim="2" count="2 2 1"
              spacing="0.8 0.8 0.8" radius="0.01" rgba="0.9 0.2 0.1 1">
      <edge equality="true"/>
      <contact selfcollide="none"/>
    </flexcomp>
  </worldbody>
</mujoco>
"""


class RenderContextTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    if not mjxw.WARP_INSTALLED:
      raise absltest.SkipTest('Warp not installed.')
    cls.previous_cache_dir = wp.config.kernel_cache_dir
    cls.tempdir = tempfile.TemporaryDirectory()
    wp.config.kernel_cache_dir = cls.tempdir.name
    cls.m = mujoco.MjModel.from_xml_string(_XML)
    graph_mode = (
        mjxw.types.GraphMode.NONE
        if jax.local_devices()[0].platform == 'cpu'
        else mjxw.types.GraphMode.WARP_STAGED
    )
    cls.mx = mjx.put_model(cls.m, impl='warp', graph_mode=graph_mode)
    cls.assets = mjx.put_render_assets(
        cls.m,
        cam_res=(16, 16),
        render_rgb=True,
        render_depth=True,
        render_seg=True,
        use_shadows=False,
    )
    cls.poses = []
    for xpos in (-0.45, -0.15, 0.15, 0.45):
      d = mujoco.MjData(cls.m)
      d.qpos[0] = xpos
      mujoco.mj_forward(cls.m, d)
      cls.poses.append(mjx.put_data(cls.m, d, impl='warp'))

  @classmethod
  def tearDownClass(cls):
    if hasattr(cls, 'tempdir'):
      wp.config.kernel_cache_dir = cls.previous_cache_dir
      cls.tempdir.cleanup()
    super().tearDownClass()

  def _frame(self, d):
    ctx = mjx.make_render_context(self.assets)
    return mjx.render_with_segmentation(self.mx, d, ctx)[:3]

  def _stack_worlds(self, poses):
    return jax.tree.map_with_path(
        lambda path, *x: (
            x[0]
            if types.tree_path_to_attr_str(path) in mjxw.types.DATA_NON_VMAP
            else jp.stack(x)
        ),
        *poses,
    )

  def _batch(self, count):
    return self._stack_worlds(self.poses[:count])

  def _nested_batch(self):
    return self._stack_worlds(
        (self._stack_worlds(self.poses[:2]), self._stack_worlds(self.poses[2:]))
    )

  def _reference(self, count):
    frame = jax.jit(self._frame)
    return jax.tree.map(
        lambda *x: np.stack(x),
        *(frame(d) for d in self.poses[:count]),
    )

  def _assert_images_equal(self, actual, expected):
    self.assertLen(actual, 3)
    for image, reference in zip(actual, expected):
      np.testing.assert_array_equal(image, reference)

  def test_vmap_construction_and_batch_sizes(self):
    frame = jax.jit(jax.vmap(self._frame))
    for count in (1, 4, 2):
      with self.subTest(count=count):
        expected = self._reference(count)
        actual = frame(self._batch(count))
        self._assert_images_equal(actual, expected)
        self.assertEqual(actual[0].shape, (count, 16 * 16))
        self.assertEqual(actual[1].shape, (count, 16 * 16))
        self.assertEqual(actual[2].shape, (count, 16 * 16, 2))
        self.assertGreater(np.count_nonzero(actual[1]), 0)
        self.assertGreater(np.unique(actual[1]).size, 1)
        self.assertTrue(np.any(np.asarray(actual[2])[..., 0] == 1))
        if count > 1:
          self.assertFalse(np.array_equal(actual[0][0], actual[0][-1]))
          self.assertFalse(np.array_equal(actual[1][0], actual[1][-1]))

  def test_nested_vmap(self):
    data = self._nested_batch()
    actual = jax.jit(jax.vmap(jax.vmap(self._frame)))(data)
    expected = tuple(x.reshape(2, 2, *x.shape[1:]) for x in self._reference(4))
    self._assert_images_equal(actual, expected)

  def test_nested_vmap_depth_only(self):
    assets = mjx.put_render_assets(
        self.m,
        cam_res=(16, 16),
        render_rgb=False,
        render_depth=True,
        render_seg=False,
        use_shadows=False,
    )

    def frame(d):
      ctx = mjx.make_render_context(assets)
      return mjx.render(self.mx, d, ctx)[:2]

    rgb, depth = jax.jit(jax.vmap(jax.vmap(frame)))(self._nested_batch())
    self.assertEqual(rgb.shape, (2, 2, 0))
    expected = self._reference(4)[1].reshape(2, 2, 16 * 16)
    np.testing.assert_array_equal(depth, expected)

  def test_matches_legacy_context(self):
    owner = mjx.create_render_context(
        self.m,
        nworld=1,
        cam_res=(16, 16),
        render_rgb=True,
        render_depth=True,
        render_seg=True,
        use_shadows=False,
    )

    def legacy_frame(d):
      d = mjx.refit_bvh(self.mx, d, owner.pytree())
      images = mjx.render_with_segmentation(self.mx, d, owner.pytree())[:3]
      return tuple(x[0] for x in images)

    expected = jax.tree.map(
        lambda *x: np.stack(x),
        *(jax.jit(legacy_frame)(d) for d in self.poses),
    )
    actual = jax.jit(jax.vmap(self._frame))(self._batch(4))
    self._assert_images_equal(actual, expected)

  def test_flex_antialiasing_matches_legacy_context(self):
    m = mujoco.MjModel.from_xml_string(_FLEX_XML)
    mx = mjx.put_model(
        m, impl='warp', graph_mode=self.mx.opt._impl.graph_mode
    )
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    base = mjx.put_data(m, d, impl='warp')
    poses = []
    for xpos, height in ((-0.3, -0.2), (0.3, 0.2)):
      vertices = np.array(d.flexvert_xpos)
      vertices[:, 0] += xpos
      vertices[0, 2] += height
      poses.append(
          base.tree_replace({'_impl.flexvert_xpos': jp.array(vertices)})
      )
    options = dict(
        cam_res=(16, 16),
        render_rgb=True,
        render_depth=True,
        render_seg=True,
        samples_per_pixel=2,
        use_shadows=False,
    )
    assets = mjx.put_render_assets(m, **options)
    owner = mjx.create_render_context(m, nworld=1, **options)

    def frame(d):
      ctx = mjx.make_render_context(assets)
      return mjx.render_with_segmentation(mx, d, ctx)[:3]

    def legacy_frame(d):
      d = mjx.refit_bvh(mx, d, owner.pytree())
      images = mjx.render_with_segmentation(mx, d, owner.pytree())[:3]
      return tuple(x[0] for x in images)

    expected = jax.tree.map(
        lambda *x: np.stack(x), *(jax.jit(legacy_frame)(d) for d in poses)
    )
    batched_frame = jax.jit(jax.vmap(frame))
    actual = batched_frame(self._stack_worlds(poses))
    self._assert_images_equal(actual, expected)
    self.assertFalse(np.array_equal(actual[0][0], actual[0][1]))
    self.assertFalse(np.array_equal(actual[1][0], actual[1][1]))
    reversed_images = batched_frame(self._stack_worlds(poses[::-1]))
    self._assert_images_equal(reversed_images, tuple(x[::-1] for x in expected))

  def test_scan_construction(self):
    def frame(carry, geom_xpos):
      d = self.poses[0].replace(geom_xpos=geom_xpos)
      ctx = mjx.make_render_context(self.assets)
      rgb, depth, seg, d = mjx.render_with_segmentation(self.mx, d, ctx)
      return carry + 1, (rgb, depth, seg)

    scan = jax.jit(lambda positions: jax.lax.scan(frame, 0, positions))
    count, actual = scan(jp.stack([d.geom_xpos for d in self.poses]))
    self.assertEqual(count, 4)
    self._assert_images_equal(actual, self._reference(4))

  def test_pmap(self):
    devices = jax.local_devices()
    if len(devices) < 2:
      self.skipTest('Requires multiple local JAX devices.')
    data = jax.tree.map(lambda *x: np.stack(x), *self.poses[:2])
    actual = jax.pmap(self._frame, devices=devices[:2])(data)
    self._assert_images_equal(actual, self._reference(2))

  def test_pmap_vmap_local_worlds(self):
    devices = jax.local_devices()
    if len(devices) < 2:
      self.skipTest('Requires multiple local JAX devices.')
    data = jax.tree.map(
        lambda *x: np.stack(x),
        self._stack_worlds(self.poses[:2]),
        self._stack_worlds(self.poses[2:]),
    )
    actual = jax.pmap(jax.vmap(self._frame), devices=devices[:2])(
        jax.device_get(data)
    )
    expected = tuple(x.reshape(2, 2, *x.shape[1:]) for x in self._reference(4))
    self._assert_images_equal(actual, expected)

  def test_vmap_model_with_unmapped_data(self):
    colors = jp.array([[0.9, 0.1, 0.1, 1.0], [0.1, 0.9, 0.1, 1.0]])

    def frame(color):
      mx = self.mx.replace(geom_rgba=self.mx.geom_rgba.at[1].set(color))
      ctx = mjx.make_render_context(self.assets)
      return mjx.render_with_segmentation(mx, self.poses[0], ctx)[:3]

    actual = jax.jit(jax.vmap(frame))(colors)
    expected = jax.tree.map(
        lambda *x: np.stack(x), *(jax.jit(frame)(c) for c in colors)
    )
    self._assert_images_equal(actual, expected)
    self.assertFalse(np.array_equal(actual[0][0], actual[0][1]))

  def test_context_lifetime_and_unpacking(self):
    assets = mjx.put_render_assets(
        self.m,
        cam_res=(16, 16),
        render_rgb=True,
        render_depth=True,
        render_seg=True,
        use_shadows=False,
    )
    factory = jax.jit(lambda: mjx.make_render_context(assets))
    ctx = factory()
    del factory, assets
    gc.collect()
    rgb, depth, seg, result_data = jax.jit(mjx.render_with_segmentation)(
        self.mx, self.poses[0], ctx
    )
    self._assert_images_equal((rgb, depth, seg), self._frame(self.poses[0]))
    for actual, expected in zip(
        jax.tree.leaves(result_data), jax.tree.leaves(self.poses[0])
    ):
      np.testing.assert_array_equal(actual, expected)
    unpacked_rgb = jax.jit(lambda x: mjx.get_rgb(ctx, 0, x))(rgb)
    unpacked_depth = jax.jit(lambda x: mjx.get_depth(ctx, 0, x, 4.0))(depth)
    unpacked_seg = jax.jit(lambda x: mjx.get_segmentation(ctx, 0, x))(seg)
    self.assertEqual(unpacked_rgb.shape, (16, 16, 3))
    self.assertEqual(unpacked_depth.shape, (16, 16, 1))
    self.assertEqual(unpacked_seg.shape, (16, 16))
    np.testing.assert_array_equal(unpacked_seg, seg[:, 0].reshape(16, 16))
    np.testing.assert_array_equal(
        unpacked_depth,
        np.clip(np.asarray(depth) / 4.0, 0, 1).reshape(16, 16, 1),
    )

  def test_render_returns_unchanged_data(self):
    ctx = mjx.make_render_context(self.assets)
    rgb, depth, result_data = jax.jit(mjx.render)(self.mx, self.poses[0], ctx)
    expected = self._frame(self.poses[0])
    np.testing.assert_array_equal(rgb, expected[0])
    np.testing.assert_array_equal(depth, expected[1])
    for actual, reference in zip(
        jax.tree.leaves(result_data), jax.tree.leaves(self.poses[0])
    ):
      np.testing.assert_array_equal(actual, reference)

  def test_compiled_executable_retains_and_releases_assets(self):
    def compile_frame():
      assets = mjx.put_render_assets(
          self.m,
          cam_res=(16, 16),
          render_rgb=True,
          render_depth=True,
          render_seg=True,
          use_shadows=False,
      )

      def frame(d):
        ctx = mjx.make_render_context(assets)
        return mjx.render_with_segmentation(self.mx, d, ctx)[:3]

      return (
          jax.jit(frame).lower(self.poses[0]).compile(),
          weakref.ref(assets),
      )

    compiled, reference = compile_frame()
    jax.clear_caches()
    gc.collect()
    self.assertIsNotNone(reference())
    actual = jax.device_get(compiled(self.poses[0]))
    self._assert_images_equal(actual, self._frame(self.poses[0]))
    del compiled
    jax.clear_caches()
    gc.collect()
    self.assertIsNone(reference())

  def test_concurrent_execution(self):
    compiled = jax.jit(jax.vmap(self._frame)).lower(self._batch(2)).compile()
    first = self._batch(2)
    second = self._stack_worlds(self.poses[2:])
    expected = self._reference(4)

    def run(data):
      return jax.device_get(compiled(data))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
      for _ in range(3):
        futures = [executor.submit(run, data) for data in (first, second)]
        for i, future in enumerate(futures):
          self._assert_images_equal(
              future.result(), tuple(x[i * 2 : (i + 1) * 2] for x in expected)
          )


if __name__ == '__main__':
  absltest.main()
