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
"""Rendering asset ownership for compiled FFI calls."""

from jax.extend import core
from jax.interpreters import batching
from jax.interpreters import mlir

_keepalive_p = core.Primitive('mjx_render_keepalive')
_keepalive_p.multiple_results = True
_keepalive_p.def_impl(lambda *args, assets: args)
_keepalive_p.def_abstract_eval(lambda *avals, assets: avals)


def _keepalive_lowering(ctx, *args, assets):
  ctx.module_context.add_keepalive(assets)
  return args


mlir.register_lowering(_keepalive_p, _keepalive_lowering)
batching.primitive_batchers[_keepalive_p] = lambda args, dims, assets: (
    _keepalive_p.bind(*args, assets=assets),
    dims,
)


def keepalive(outputs, assets):
  """Retains `assets` through the executable that produces `outputs`."""
  return _keepalive_p.bind(*outputs, assets=assets)
