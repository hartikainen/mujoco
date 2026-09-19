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
"""Tests for MJX-Warp FFI helpers."""

import inspect
import os
import tempfile
from unittest import mock

from absl.testing import absltest
import jax
from mujoco.mjx._src import io
import mujoco.mjx.warp as mjxw
import numpy as np


try:
  from mujoco.mjx.warp import ffi  # pylint: disable=g-import-not-at-top
except ImportError:
  ffi = None

_FORCE_TEST = os.environ.get('MJX_WARP_FORCE_TEST', '0') == '1'


class FfiTest(absltest.TestCase):

  def test_empty_vector_operand_signature(self):
    if not mjxw.WARP_INSTALLED:
      self.skipTest('Warp not installed.')

    restored = mock.sentinel.empty
    output = mock.sentinel.output
    received = []

    def func(
        values: ffi.wp.array[float],
        empty: ffi.wp.array2d[ffi.wp.vec3],
        result: ffi.wp.array[float],
    ):
      received.append((values, empty, result))

    def create_callable(callback, **kwargs):
      del kwargs
      self.assertEqual(
          tuple(inspect.signature(callback).parameters), ('values', 'result')
      )
      return lambda *args: callback(*args, output)

    with mock.patch.object(ffi, '_JAX_CALLABLE_VARIADIC_TUPLE_REGISTRY', {}):
      with mock.patch.object(ffi.wp, 'jax_callable', create_callable):
        with mock.patch.object(ffi.wp, 'empty', return_value=restored) as alloc:
          call = ffi.jax_callable_variadic_tuple(
              func, output_dims={'result': (3,)}
          )
          values = np.ones(3, dtype=np.float32)
          call(values, np.empty((1, 0, 3), dtype=np.float32))
          alloc.assert_called_once_with(shape=(1, 0), dtype=ffi.wp.vec3)
          self.assertLen(received, 1)
          self.assertIs(received[0][0], values)
          self.assertIs(received[0][1], restored)
          self.assertIs(received[0][2], output)

  def test_pmap_empty_operand(self):
    if not mjxw.WARP_INSTALLED:
      self.skipTest('Warp not installed.')
    devices = jax.local_devices()
    if len(devices) < 2:
      self.skipTest('Requires multiple local JAX devices.')

    def copy_values(
        values: ffi.wp.array[float],
        empty: ffi.wp.array[float],
        output: ffi.wp.array[float],
    ):
      del empty
      ffi.wp.copy(output, values)

    with tempfile.TemporaryDirectory() as cache_dir:
      with mock.patch.object(ffi.wp.config, 'kernel_cache_dir', cache_dir):
        call = ffi.jax_callable_variadic_tuple(
            copy_values,
            graph_mode=ffi.wp.JaxCallableGraphMode.NONE,
            output_dims={'output': (3,)},
        )
        values = np.arange(6, dtype=np.float32).reshape(2, 3)
        empty = np.empty((2, 0), dtype=np.float32)
        result = jax.pmap(jax.jit(call), devices=devices[:2])(values, empty)[0]
        np.testing.assert_array_equal(result, values)

  def test_jax_callable_variadic_tuple_cache(self):
    if not _FORCE_TEST:
      if not mjxw.WARP_INSTALLED:
        self.skipTest('Warp not installed.')
      if not io.has_cuda_gpu_device():
        self.skipTest('No CUDA GPU device available.')

    def func_a(values: tuple[int, ...], output: int):
      del values, output

    def func_b(values: tuple[int, ...], output: int):
      del values, output

    # Count callback registrations without creating real JAX FFI targets.
    def create_callable(*args, **kwargs):
      del args, kwargs
      return mock.Mock()

    with mock.patch.object(ffi, '_JAX_CALLABLE_VARIADIC_TUPLE_REGISTRY', {}):
      with mock.patch.object(
          ffi.wp, 'jax_callable', side_effect=create_callable
      ) as jax_callable:
        wrapper_a = ffi.jax_callable_variadic_tuple(func_a)
        wrapper_a_again = ffi.jax_callable_variadic_tuple(func_a)
        wrapper_b = ffi.jax_callable_variadic_tuple(func_b)

        # Equivalent traces of the same callable reuse one callback.
        wrapper_a((1, 2))
        wrapper_a_again((3, 4))
        self.assertEqual(jax_callable.call_count, 1)

        # Distinct shim functions with the same signature do not share a target.
        wrapper_b((5, 6))
        self.assertEqual(jax_callable.call_count, 2)

        # A different tuple arity changes the signature and PyTree cache key.
        wrapper_b((1, 2, 3))
        self.assertEqual(jax_callable.call_count, 3)


if __name__ == '__main__':
  absltest.main()
