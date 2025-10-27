# Copyright 2023 DeepMind Technologies Limited
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
"""Render + RenderContextRegistry helpers."""

from typing import Any
import flax
import threading
# pylint: disable=g-importing-member
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import Impl
from mujoco.mjx._src.types import Model
import mujoco
# pylint: enable=g-importing-member
import mujoco.mjx.warp as mjxw


def render(m: Model, d: Data, ctx: Any) -> Data:
  """Render."""
  if m.impl == Impl.WARP and d.impl == Impl.WARP and mjxw.WARP_INSTALLED:
    from mujoco.mjx.warp import render as mjxw_render  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    d = d.replace(_render_context=ctx)
    return mjxw_render.render(m, d)

  raise NotImplementedError(f'render only implemented for Warp backend.')


def create_render_context(
    mjm: mujoco.MjModel,
    m: Model,
    d: Data,
    nworld: int,
    width: int,
    height: int,
    use_textures: bool,
    use_shadows: bool,
    fov_rad: float,
    render_rgb: bool,
    render_depth: bool,
    enabled_geom_groups = [0, 1, 2],
):
  """Render."""
  if m.impl == Impl.WARP and d.impl == Impl.WARP and mjxw.WARP_INSTALLED:
    from mujoco.mjx.warp import render as mjxw_render  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    return mjxw_render.create_render_context_in_registry(
      mjm,
      nworld,
      width,
      height,
      use_textures,
      use_shadows,
      fov_rad,
      render_rgb,
      render_depth,
      enabled_geom_groups,
    )

  raise NotImplementedError(f'render only implemented for Warp backend.')