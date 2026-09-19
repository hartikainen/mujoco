The MJX renderer supports constructing a render context inside
`jax.{jit,vmap,pmap,scan}` without specifying `nworld`. Model-dependent assets
are prepared before tracing; mutable rendering workspaces are allocated for
the device and batch shape of each compiled render call.

The public API separates these responsibilities:

- `mjx.put_render_assets(mjm, **settings)` prepares shared assets, including
  camera layouts, textures, and mesh acceleration structures. Model and
  rendering settings are captured at preparation time.
- `mjx.make_render_context(assets)` constructs a lightweight context value
  inside or outside JAX transformations. The value owns no per-world buffers.
- `mjx.render` and `mjx.render_with_segmentation` accept this context and fuse
  scene acceleration structure refitting with rendering. Their return tuples
  retain the input data, and image batch dimensions follow the transformations.
  A separate `mjx.refit_bvh` call is unnecessary for this context.

For example, context construction can live inside the transformed function:

```python
assets = mjx.put_render_assets(mjm, render_rgb=True, render_depth=True)

def frame(d):
    context = mjx.make_render_context(assets)
    rgb, depth, _ = mjx.render(mx, d, context)
    return rgb, depth

render_batch = jax.jit(jax.vmap(frame))
render_devices = jax.pmap(frame)
```

The example assumes `mx` is a Warp-backed MJX model and inputs have the
appropriate mapped axes. Nested `vmap` and `pmap(vmap(...))` infer each local
workspace's world count. The explicit `mjx.create_render_context` API remains
available for callers that supply `nworld`.

The vendored Warp renderer separates immutable assets from scene and flex
acceleration structures, antialiasing scratch space, and output buffers.
Workspaces are cached by device and input signature. CUDA execution stages
inputs in stable allocations for internal graph replay. A host lock and stream
events order workspace reuse after output copies. Compiled executables retain
asset ownership through a JAX lowering keepalive, so dropping Python references
does not invalidate executable resources.

The FFI binding omits read-only empty operands when output dimensions are
explicit, reconstructs their Warp arrays in the callback, and preserves
in-place operands. Empty-array specifications participate in the callback
cache key. This avoids the observed CPU compilation failure for mapped empty
operands. The generated binding also uses an inline `jit`, and output reshaping
handles empty image channels without inferring an ambiguous dimension. The
generator and renderer example use the functional context API.

The implementation is committed in dependency order:

- `01b79060ced7`: Separate Warp render assets from per-world workspace.
- `41890bbf2650`: Handle empty operands in MJX Warp FFI batching.
- `1dcaa5472da9`: Make MJX render contexts independent of batch size.

Validation for these commits passes on CPU with forced host devices:

- `mujoco.mjx.warp.render_context_test` covers `vmap`, nested `vmap`, `pmap`,
  `pmap(vmap(...))`, context construction in `scan`, and mapped models with
  unmapped data. It also checks legacy image equivalence, flex deformation with
  antialiasing, depth-only output, concurrent execution, and asset lifetime.
- `mujoco.mjx.warp.ffi_test` covers empty vector operands, callback caching,
  and empty operands through `pmap(jit(...))`.
- `mujoco.mjx._src.render_util_test` passes. Generated binding regeneration is
  byte-identical; Python compilation and `git diff --check` also pass.

The renderer transformation checks can be reproduced with:

```sh
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  .venv/bin/python -m mujoco.mjx.warp.render_context_test
```

Run the FFI and render utility modules with the same environment and
`MJX_WARP_FORCE_TEST=1`. Validation logs are in
`/tmp/mjx_render_context_full.log`, `/tmp/mjx_ffi_final.log`, and
`/tmp/mjx-render-util-tests.log`.

CUDA hardware is unavailable in the validation environment. CUDA graph replay,
GPU event ordering, and execution across GPUs remain unverified; the next check
is to run the transformation tests on CUDA devices. Splat assets require a
shared group selection in the functional API.

The implementation commits are local and have not been pushed. Unrelated
untracked workspace files are untouched.
