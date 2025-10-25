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
    geom_dataid: wp.array(dtype=int),
    geom_matid: wp.array2d(dtype=int),
    geom_rgba: wp.array2d(dtype=wp.vec4),
    geom_size: wp.array2d(dtype=wp.vec3),
    geom_type: wp.array(dtype=int),
    light_active: wp.array2d(dtype=bool),
    light_castshadow: wp.array2d(dtype=bool),
    light_type: wp.array2d(dtype=int),
    mat_rgba: wp.array2d(dtype=wp.vec4),
    mat_texid: wp.array3d(dtype=int),
    mat_texrepeat: wp.array2d(dtype=wp.vec2),
    mesh_face: wp.array(dtype=wp.vec3i),
    mesh_faceadr: wp.array(dtype=int),
    ncam: int,
    ngeom: int,
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
  _m.nlight = nlight
  _d.cam_xmat = cam_xmat
  _d.cam_xpos = cam_xpos
  _d.geom_xmat = geom_xmat
  _d.geom_xpos = geom_xpos
  _d.light_xdir = light_xdir
  _d.light_xpos = light_xpos
  mjwarp.render(_m, _d, rc_id)


def _render_jax_impl(m: types.Model, d: types.Data, rc_id: int):
  jf = ffi.jax_callable_variadic_tuple(
      _render_shim,
      num_outputs=0,
      vmap_method=None,
  )
  jf(
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
      m.nlight,
      d.cam_xmat,
      d.cam_xpos,
      d.geom_xmat,
      d.geom_xpos,
      d._impl.light_xdir,
      d._impl.light_xpos,
      int(rc_id),
  )
  return d


@jax.custom_batching.custom_vmap
@ffi.marshal_jax_warp_callable
def render(m: types.Model, d: types.Data, rc_id: int):
  return _render_jax_impl(m, d, rc_id)


@render.def_vmap
@ffi.marshal_custom_vmap
def render_vmap(unused_axis_size, is_batched, m, d, rc_id):
  d = render(m, d, rc_id)
  return d, is_batched[1]