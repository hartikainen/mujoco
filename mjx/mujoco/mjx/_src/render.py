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
"""Render helpers for MJX."""

from typing import Any

import jax
import mujoco.mjx.warp as mjxw

# pylint: disable=g-importing-member
from mujoco.mjx._src.types import Data
from mujoco.mjx._src.types import Impl
from mujoco.mjx._src.types import Model

# pylint: enable=g-importing-member


def _require_segmentation_enabled(warp_rc) -> None:
  """Raises if the render context has no segmentation-enabled cameras."""
  if not (warp_rc.seg_adr.numpy() >= 0).any():
    raise ValueError(
        'Render context was not configured with segmentation rendering. '
        'Pass render_seg=True or enable it for at least one camera in '
        'create_render_context.'
    )


def _call_render(
    m: Model,
    d: Data,
    ctx: Any,
    require_seg: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, Data]:
  if m.impl == Impl.WARP and d.impl == Impl.WARP and mjxw.WARP_INSTALLED:
    from mujoco.mjx.warp import render as mjxw_render  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    from mujoco.mjx.warp import render_context  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error

    warp_rc = render_context.get(ctx)
    if require_seg:
      _require_segmentation_enabled(warp_rc)
    if isinstance(ctx, render_context.RenderContextValue):
      from mujoco.mjx.warp import render_frame  # pylint: disable=g-import-not-at-top

      rgb, depth, seg = render_frame.render_frame(m, d, ctx)
      return rgb, depth, seg, d
    rgb, depth, seg, token_array = mjxw_render.render(m, d, ctx)
    token = token_array.reshape(
        d._impl._jax_token.shape  # pytype: disable=attribute-error
    )
    d = d.tree_replace(
        {'_impl._jax_token': token}
    )  # pyrefly: ignore[bad-assignment]
    return rgb, depth, seg, d

  raise NotImplementedError('render only implemented for MuJoCo Warp.')


def render(m: Model, d: Data, ctx: Any) -> tuple[jax.Array, jax.Array, Data]:
  """Render packed RGB and depth buffers.

  Contexts from `make_render_context` refit acceleration structures internally.
  Their outputs preserve mapped axes and have no implicit world axis. Their
  returned `Data` is unchanged.

  Returns:
    `(rgb, depth, d)` with packed image buffers. Legacy contexts return `d`
    with an execution token; contexts from `make_render_context` return the
    supplied `d`.
  """
  rgb, depth, unused_seg, d = _call_render(m, d, ctx)
  return rgb, depth, d


def render_with_segmentation(
    m: Model, d: Data, ctx: Any
) -> tuple[jax.Array, jax.Array, jax.Array, Data]:
  """Render and return RGB, depth, and packed segmentation outputs.

  Returns:
    `(rgb, depth, seg, d)` with packed image buffers. Contexts from
    `make_render_context` refit internally and return the supplied `d`;
    legacy contexts return `d` with an execution token.
  """
  rgb, depth, seg, d = _call_render(m, d, ctx, require_seg=True)
  return rgb, depth, seg, d
