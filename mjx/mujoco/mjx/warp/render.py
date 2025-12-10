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

"""DO NOT EDIT. This file is auto-generated."""
import dataclasses
import jax
from mujoco.mjx._src import types
from mujoco.mjx.warp import ffi
import mujoco.mjx.third_party.mujoco_warp as mjwarp
from mujoco.mjx.third_party.mujoco_warp._src import types as mjwp_types
import warp as wp
import flax
import threading
import mujoco
import functools

_m = mjwarp.Model(
    **{f.name: None for f in dataclasses.fields(mjwarp.Model) if f.init}
)
_d = mjwarp.Data(
    **{f.name: None for f in dataclasses.fields(mjwarp.Data) if f.init}
)
_o = mjwarp.Option(
    **{f.name: None for f in dataclasses.fields(mjwarp.Option) if f.init}
)
_s = mjwarp.Statistic(
    **{f.name: None for f in dataclasses.fields(mjwarp.Statistic) if f.init}
)
_c = mjwarp.Contact(
    **{f.name: None for f in dataclasses.fields(mjwarp.Contact) if f.init}
)
_e = mjwarp.Constraint(
    **{f.name: None for f in dataclasses.fields(mjwarp.Constraint) if f.init}
)


@ffi.format_args_for_warp
def _render_shim(
    # Model
    nworld: int,
    geom_dataid: wp.array(dtype=int),
    geom_matid: wp.array2d(dtype=int),
    geom_rgba: wp.array2d(dtype=wp.vec4),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_type: wp.array(dtype=int),
    light_active: wp.array2d(dtype=bool),
    light_castshadow: wp.array2d(dtype=bool),
    light_type: wp.array2d(dtype=int),
    mat_rgba: wp.array2d(dtype=wp.vec4),
    mat_texid: wp.array2d(dtype=int),
    mat_texrepeat: wp.array2d(dtype=wp.vec2),
    mesh_face: wp.array(dtype=wp.vec3i),
    mesh_faceadr: wp.array(dtype=int),
    ncam: int,
    ngeom: int,
    nflex: int,
    nlight: int,
    # Data
    cam_xmat: wp.array2d(dtype=wp.mat33),
    cam_xpos: wp.array2d(dtype=wp.vec3),
    geom_xmat: wp.array2d(dtype=wp.mat33),
    geom_xpos: wp.array2d(dtype=wp.vec3),
    light_xdir: wp.array2d(dtype=wp.vec3),
    light_xpos: wp.array2d(dtype=wp.vec3),
    # Registry
    rc_id: int,
    device_specific_rc_id: bool,
    # Out
    rgb_out: wp.array2d(dtype=wp.uint32),
    depth_out: wp.array2d(dtype=float),
):
  _m.stat = _s
  _m.opt = _o
  _d.efc = _e
  _d.contact = _c
  _m.geom_dataid = geom_dataid
  _m.geom_matid = geom_matid
  _m.geom_rgba = geom_rgba
  _m.geom_size = geom_size
  _m.geom_type = geom_type
  _m.light_active = light_active
  _m.light_castshadow = light_castshadow
  _m.light_type = light_type
  _m.mat_rgba = mat_rgba
  _m.mat_texid = mat_texid
  _m.mat_texrepeat = mat_texrepeat
  _m.mesh_face = mesh_face
  _m.mesh_faceadr = mesh_faceadr
  _m.ncam = ncam
  _m.ngeom = ngeom
  _m.nflex = nflex
  _m.nlight = nlight
  _d.cam_xmat = cam_xmat
  _d.cam_xpos = cam_xpos
  _d.geom_xmat = geom_xmat
  _d.geom_xpos = geom_xpos
  _d.light_xdir = light_xdir
  _d.light_xpos = light_xpos
  _d.nworld = nworld

  if device_specific_rc_id:
    # TODO(hartikainen): We need to do this check under/within the ffi call. If we do it
    # outside, `wp.get_device()` always results in `cuda:0` (because that's where the
    # JAX tracing happens) and hence the device gets hard-coded to the ffi call. By
    # calling `wp.get_device()` inside the ffi call, we get the right device.
    #
    # TODO(hartikainen): This is a hacky workaround. It's really error prone, opaque,
    # and what's worse, the user needs to explicitly create/put render contexts for
    # every device.
    key = (rc_id, str(wp.get_device()))
  else:
    key = rc_id
  render_context = _RENDER_CONTEXT_BUFFERS[key]
  mjwarp.render(_m, _d, render_context)
  # TODO: avoid copy?
  wp.copy(rgb_out, render_context.rgb_data)
  wp.copy(depth_out, render_context.depth_data)


def _render_jax_impl(m: types.Model, d: types.Data):
  rc_id = d._render_context.key
  if d._device_specific_rc_id:
    key = (rc_id, str(wp.get_device()))
  else:
    key = rc_id

  render_ctx = _RENDER_CONTEXT_BUFFERS[key]

  # TODO(hartikainen): Doesn't this always evaluate to `render_ctx.ncam` because
  # `render_ctx.render_{rgb,depth}` are `wp.array`s?
  ncam_rgb = render_ctx.ncam if render_ctx.render_rgb else 0
  ncam_depth = render_ctx.ncam if render_ctx.render_depth else 0
  output_dims = {
    'rgb_out': render_ctx.rgb_data.shape,
    'depth_out': render_ctx.depth_data.shape,
  }
  jf = ffi.jax_callable_variadic_tuple(
      _render_shim,
      num_outputs=2,
      output_dims=output_dims,
      vmap_method=None,
      # TODO(hartikainen): `in_out_argnames = {"rgb_out", "depth_out"}`?
  )

  out = jf(
      d.qpos.shape[0],
      m.geom_dataid,
      m.geom_matid,
      m.geom_rgba,
      m.geom_size,
      m.geom_type,
      m._impl.light_active,
      m.light_castshadow,
      m.light_type,
      m.mat_rgba,
      m.mat_texid,
      m._impl.mat_texrepeat,
      m.mesh_face,
      m.mesh_faceadr,
      m.ncam,
      m.ngeom,
      m._impl.nflex,
      m.nlight,
      d.cam_xmat,
      d.cam_xpos,
      d.geom_xmat,
      d.geom_xpos,
      d._impl.light_xdir,
      d._impl.light_xpos,
      d._render_context.key,
      d._device_specific_rc_id,
  )
  return out


@jax.custom_batching.custom_vmap
@functools.partial(ffi.marshal_jax_warp_callable, raw_output=True)
def render(m: types.Model, d: types.Data):
  return _render_jax_impl(m, d)


@render.def_vmap
@functools.partial(ffi.marshal_custom_vmap, raw_output=True)
def render_vmap(unused_axis_size, is_batched, m, d):
  out = render(m, d)
  return out, [True, True]



_RENDER_CONTEXT_BUFFERS = {}
_RENDER_CONTEXT_LOCK = threading.Lock()


@flax.struct.dataclass(frozen=True)
class RenderContextRegistry:
  key: str | int | tuple[str | int, ...]

  # NOTE(hartikainen): I've commented this out because we discard the default render
  # context keys and hence this destructor deletes the render contexts too early.

  # def __del__(self):
  #   lock = globals().get('_RENDER_CONTEXT_LOCK')
  #   buffers = globals().get('_RENDER_CONTEXT_BUFFERS')
  #   if lock is None or buffers is None:
  #     return
  #   try:
  #     with lock:
  #       buffers.pop(self.key, None)
  #   except Exception:
  #     pass


def create_render_context(
  mjm: mujoco.MjModel,
  m: types.Model,
  d: types.Data,
  cam_res: list[tuple[int, int]] | tuple[int, int],
  render_rgb: bool,
  render_depth: bool,
  use_textures: bool,
  use_shadows: bool,
  enabled_geom_groups = [0, 1, 2],
  cam_active: list[bool] | None = None,
  flex_render_smooth: bool = True,
):
  from mujoco.mjx.third_party.mujoco_warp._src import render_context
  from mujoco.mjx.third_party import mujoco_warp as mjw

  # TODO: think of a better way besides this hack?
  # what happens if the geom_size changes later?
  mw = mjw.put_model(mjm)
  dw = mjw.make_data(mjm)

  dw.qpos = wp.array(shape=d.qpos.shape, dtype=wp.dtype_from_jax(d.qpos.dtype))
  dw.nworld = d.qpos.shape[0] if d.qpos.ndim > 1 else 1

  rc = render_context.RenderContext(
    mjm,
    mw,
    dw,
    cam_res,
    render_rgb,
    render_depth,
    use_textures,
    use_shadows,
    enabled_geom_groups,
    cam_active,
    flex_render_smooth,
  )
  return rc


def create_render_context_in_registry(
  mjm: mujoco.MjModel,
  m: types.Model,
  d: types.Data,
  cam_res: list[tuple[int, int]] | tuple[int, int],
  render_rgb: bool,
  render_depth: bool,
  use_textures: bool,
  use_shadows: bool,
  enabled_geom_groups = [0, 1, 2],
  cam_active: list[bool] | None = None,
  flex_render_smooth: bool = True,
  key: str | int | tuple[str | int, ...] | None = None,
):
  rc = create_render_context(
    mjm=mjm,
    m=m,
    d=d,
    cam_res=cam_res,
    render_rgb=render_rgb,
    render_depth=render_depth,
    use_textures=use_textures,
    use_shadows=use_shadows,
    enabled_geom_groups=enabled_geom_groups,
    cam_active=cam_active,
    flex_render_smooth=flex_render_smooth,
  )
  with _RENDER_CONTEXT_LOCK:
    if key is None:
      key = len(_RENDER_CONTEXT_BUFFERS) + 1
    assert key not in _RENDER_CONTEXT_BUFFERS, (
      f'RenderContext with key {key} already exists in registry.'
    )
    _RENDER_CONTEXT_BUFFERS[key] = rc
  return RenderContextRegistry(key)
