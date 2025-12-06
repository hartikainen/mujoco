#!/usr/bin/env sh

rsync \
  -arvz \
  --exclude 'test_data/' \
  --filter="- *_test.py" \
  --filter="- *testspeed.py" \
  ~/github/google-deepmind/mujoco_warp/mujoco_warp/ \
  mjx/mujoco/mjx/third_party/mujoco_warp/

# Fix imports and paths
find ./mjx/mujoco/mjx/third_party/mujoco_warp/_src \
  -name "*.py" \
  -print0 | \
  xargs -0 sed -i \
    -e 's/from \._src/from mujoco.mjx.third_party.mujoco_warp._src/g' \
    -e 's/from \. import/from mujoco.mjx.third_party.mujoco_warp._src import/g' \
    -e 's/from \.\([a-zA-Z_]\)/from mujoco.mjx.third_party.mujoco_warp._src.\1/g' \
    -e 's/import mujoco_warp/import mujoco.mjx.third_party.mujoco_warp/g' \
    -e 's/from mujoco_warp/from mujoco.mjx.third_party.mujoco_warp/g' \
    -e 's/epath.resource_path("mujoco_warp")/epath.resource_path("mujoco.mjx.third_party.mujoco_warp")/g'

sed -i \
    -e 's/from \._src\.\([a-zA-Z_]\)/from mujoco.mjx.third_party.mujoco_warp._src.\1/g' \
    -e 's/import mujoco_warp/import mujoco.mjx.third_party.mujoco_warp/g' \
    -e 's/from mujoco_warp/from mujoco.mjx.third_party.mujoco_warp/g' \
    ./mjx/mujoco/mjx/third_party/mujoco_warp/__init__.py

sed -i \
    -e 's/from \._src\.\([a-zA-Z_]\)/from mujoco.mjx.third_party.mujoco_warp._src.\1/g' \
    -e 's/import mujoco_warp/import mujoco.mjx.third_party.mujoco_warp/g' \
    -e 's/from mujoco_warp/from mujoco.mjx.third_party.mujoco_warp/g' \
    -e 's#epath.resource_path("mujoco_warp")#epath.resource_path("mjx") / "third_party/mujoco_warp"#g' \
    ./mjx/mujoco/mjx/third_party/mujoco_warp/viewer.py

sed -i \
    -e 's/from \._src\.\([a-zA-Z_]\)/from mujoco.mjx.third_party.mujoco_warp._src.\1/g' \
    -e 's/import mujoco_warp/import mujoco.mjx.third_party.mujoco_warp/g' \
    -e 's/from mujoco_warp/from mujoco.mjx.third_party.mujoco_warp/g' \
    ./mjx/mujoco/mjx/third_party/mujoco_warp/render.py
