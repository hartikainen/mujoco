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
"""MJX Warp render context types and buffer registry."""

import copy
import dataclasses
import functools
import itertools
import threading
import weakref

from mujoco.mjx._src import dataclasses as mjx_dataclasses

_MJX_RENDER_CONTEXT_LOCK = threading.Lock()
_MJX_RENDER_CONTEXT_BUFFERS = {}
_RENDER_ASSETS = weakref.WeakValueDictionary()
_RENDER_ASSETS_IDS = itertools.count()


def _signature(value):
  import warp as wp  # pylint: disable=g-import-not-at-top

  if isinstance(value, wp.array):
    return value.dtype, value.shape
  if dataclasses.is_dataclass(value):
    return tuple(
        _signature(getattr(value, f.name)) for f in dataclasses.fields(value)
    )
  return repr(value)


def _stage(value, copies):
  import warp as wp  # pylint: disable=g-import-not-at-top

  if isinstance(value, wp.array):
    staged = wp.empty_like(value)
    copies.append(staged)
    return staged
  if dataclasses.is_dataclass(value):
    staged = copy.copy(value)
    for field in dataclasses.fields(value):
      object.__setattr__(
          staged, field.name, _stage(getattr(value, field.name), copies)
      )
    return staged
  return value


def _arrays(value):
  import warp as wp  # pylint: disable=g-import-not-at-top

  if isinstance(value, wp.array):
    yield value
  elif dataclasses.is_dataclass(value):
    for field in dataclasses.fields(value):
      yield from _arrays(getattr(value, field.name))


class _RenderWorkspace:
  """Owns mutable rendering resources and their completion event."""

  def __init__(self, model, assets, m, d):
    from mujoco.mjx.third_party.mujoco_warp._src import render_util  # pylint: disable=g-import-not-at-top
    import warp as wp  # pylint: disable=g-import-not-at-top

    self.context = render_util.create_render_workspace(model, assets, d.nworld)
    self.inputs = []
    self.model = _stage(m, self.inputs)
    self.data = _stage(d, self.inputs)
    self.graph = None
    self.done = wp.Event() if wp.get_device().is_cuda else None
    self.pending = False

  def render(self, m, d, outputs, use_cuda_graph):
    from mujoco.mjx.third_party import mujoco_warp as mjwarp  # pylint: disable=g-import-not-at-top
    import warp as wp  # pylint: disable=g-import-not-at-top

    if self.pending:
      wp.wait_event(self.done)
    try:
      for dst, src in zip(self.inputs, (*_arrays(m), *_arrays(d))):
        wp.copy(dst, src)
      if use_cuda_graph and wp.get_device().is_cuda:
        if self.graph is None:
          with wp.ScopedCapture() as capture:
            mjwarp.refit_bvh(self.model, self.data, self.context)
            mjwarp.render(self.model, self.data, self.context)
          self.graph = capture.graph
        wp.capture_launch(self.graph)
      else:
        mjwarp.refit_bvh(self.model, self.data, self.context)
        mjwarp.render(self.model, self.data, self.context)
      for dst, src in zip(
          outputs,
          (
              self.context.rgb_data,
              self.context.depth_data,
              self.context.seg_data,
          ),
      ):
        wp.copy(dst, src)
    finally:
      if self.done is not None:
        wp.record_event(self.done)
        self.pending = True

  def __del__(self):
    if getattr(self, 'pending', False):
      try:
        import warp as wp  # pylint: disable=g-import-not-at-top
        wp.synchronize_event(self.done)
      except (ImportError, AttributeError, TypeError):
        pass


class RenderAssets:
  """Owns immutable assets and serializes device workspace reuse.

  Compiled executables retain this owner through their execution lifetime.
  CUDA events order reuse through device completion, including output copies.
  """

  def __init__(self, model, device=None, **kwargs):
    from mujoco.mjx.third_party.mujoco_warp._src import render_util  # pylint: disable=g-import-not-at-top
    import warp as wp  # pylint: disable=g-import-not-at-top

    self._model = copy.copy(model)
    self._options = copy.deepcopy(kwargs)
    self._lock = threading.RLock()
    self._bound = {}
    self._workspaces = {}
    self._devices = {}
    with wp.ScopedDevice(device):
      self.metadata = render_util.create_render_assets(
          self._model, **self._options
      )
      ready = wp.record_event() if wp.get_device().is_cuda else None
      self._devices[wp.get_device().alias] = self.metadata, ready
    with _MJX_RENDER_CONTEXT_LOCK:
      self.key = next(_RENDER_ASSETS_IDS)
      _RENDER_ASSETS[self.key] = self

  def bind(self, function):
    """Binds an FFI callback without retaining resources in its global registry."""
    with self._lock:
      if function not in self._bound:
        reference = weakref.ref(self)

        @functools.wraps(function)
        def bound(*args, **kwargs):
          owner = reference()
          if owner is None:
            raise RuntimeError('Rendering assets have been released.')
          with owner._lock:
            return function(*args, **kwargs)

        self._bound[function] = bound
      return self._bound[function]

  def render(self, m, d, outputs, use_cuda_graph):
    from mujoco.mjx.third_party.mujoco_warp._src import render_util  # pylint: disable=g-import-not-at-top
    import warp as wp  # pylint: disable=g-import-not-at-top

    device = wp.get_device()
    with self._lock:
      if device.alias not in self._devices:
        assets = render_util.create_render_assets(self._model, **self._options)
        ready = wp.record_event() if device.is_cuda else None
        self._devices[device.alias] = assets, ready
      assets, ready = self._devices[device.alias]
      if ready is not None:
        wp.wait_event(ready)
      key = device.alias, _signature(m), _signature(d), use_cuda_graph
      if key not in self._workspaces:
        self._workspaces[key] = _RenderWorkspace(self._model, assets, m, d)
      self._workspaces[key].render(m, d, outputs, use_cuda_graph)


class RenderContextValue(mjx_dataclasses.PyTreeNode):
  """Batch-independent render configuration retaining its immutable assets."""

  assets: RenderAssets


def render_frame(key, m, d, rgb, depth, seg, use_cuda_graph):
  """Renders with workspace owned by the FFI asset reference."""
  _RENDER_ASSETS[key].render(m, d, (rgb, depth, seg), use_cuda_graph)


class RenderContext:
  """MJX render context wrapping one or more warp render contexts.

  Returned by ``io.create_render_context``.  Holds the raw warp contexts
  directly so callers can read camera resolution, buffer addresses, etc.

  Use :meth:`pytree` to obtain the lightweight JAX-compatible handle
  that should be passed into ``jit``/``vmap``-compiled functions such as
  ``render`` and ``refit_bvh``.
  """

  def __init__(self, key, contexts, default):
    self.key = key
    self._contexts = contexts  # {device_ordinal: warp RenderContext}
    self._default = default  # the first warp RenderContext

  def pytree(self):
    """Returns a lightweight JAX pytree for use in jit/vmap."""
    return RenderContextPytree(self.key)

  def __getattr__(self, name):
    """Delegate attribute access to the default warp context."""
    return getattr(self._default, name)

  def __del__(self):
    lock = _MJX_RENDER_CONTEXT_LOCK
    buffers = _MJX_RENDER_CONTEXT_BUFFERS
    if lock is None or buffers is None:
      return
    with lock:
      keys_to_remove = [
          k for k in buffers.keys() if isinstance(k, tuple) and k[0] == self.key
      ]
      for k in keys_to_remove:
        buffers.pop(k, None)


class RenderContextPytree(mjx_dataclasses.PyTreeNode):
  """Minimal JAX pytree holding just the render context key.

  The key is static (aux_data) so JAX doesn't trace it, allowing the
  Warp FFI to receive a concrete int value.
  """

  key: int


def get(rc: RenderContextPytree):
  """Validates and returns the backing Warp render context."""
  if isinstance(rc, RenderContextValue):
    return rc.assets.metadata
  if not isinstance(rc, RenderContextPytree):
    raise TypeError(
        f'Expected RenderContextPytree, got {type(rc).__name__}.'
        ' Use rc.pytree() to get the JAX-compatible handle.'
    )
  return _MJX_RENDER_CONTEXT_BUFFERS[(rc.key, None)]
