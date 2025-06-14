Need to install clang:
```
sudo apt install -y clang clang-format
```

```
   bazel build --config=linux_gcc_x86_64 //...
INFO: Analyzed 203 targets (85 packages loaded, 12417 targets configured).
ERROR: /home/hartikainen/github/google-deepmind/mujoco/bazel-build/src/engine/BUILD.bazel:6:14: Compiling src/engine/engine_print.c failed: (Exit 1): gcc failed: error executing CppCompile command (from target //src/engine:engine) /usr/bin/gcc -U_FORTIFY_SOURCE -fstack-protector -Wall -Wunused-but-set-parameter -Wno-free-nonheap-object -fno-omit-frame-pointer -MD -MF ... (remaining 46 arguments skipped)

Use --sandbox_debug to see verbose messages from the sandbox and retain the sandbox build root for debugging
src/engine/engine_print.c: In function 'memorySize':
src/engine/engine_print.c:179:45: error: ' bytes' directive output may be truncated writing 6 bytes into a region of size between 0 and 15 [-Werror=format-truncation=]
  179 |     snprintf(message, sizeof(message), "%5zu bytes", nbytes);
      |                                             ^~~~~~
src/engine/engine_print.c:179:5: note: 'snprintf' output between 12 and 27 bytes into a destination of size 20
  179 |     snprintf(message, sizeof(message), "%5zu bytes", nbytes);
      |     ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cc1: all warnings being treated as errors
Use --verbose_failures to see the command lines of failed build steps.
INFO: Elapsed time: 35.298s, Critical Path: 22.70s
INFO: 423 processes: 220 internal, 203 processwrapper-sandbox.
ERROR: Build did NOT complete successfully
```


```
   bazel build --sandbox_debug //...
WARNING: Build options --action_env and --platforms have changed, discarding analysis cache (this can be expensive, see https://bazel.build/advanced/performance/iteration-speed).
INFO: Analyzed 203 targets (142 packages loaded, 13042 targets configured).
ERROR: /home/hartikainen/github/google-deepmind/mujoco/bazel-build/python/mujoco/BUILD.bazel:252:17: Compiling python/mujoco/rollout.cc failed: (Exit 1): process-wrapper failed: error executing CppCompile command
  (cd /home/hartikainen/.cache/bazel/_bazel_hartikainen/114ae8b49ccfc70a3ada0f2ba0fbb8a0/sandbox/processwrapper-sandbox/491/execroot/mujoco && \
  exec env - \
    CC=clang \
    PATH=/home/hartikainen/.cache/bazelisk/downloads/sha256/c97f02133adce63f0c28678ac1f21d65fa8255c80429b588aeeba8a1fac6202b/bin:/home/hartikainen/.local/bin:/home/hartikainen/conda/bin:/home/hartikainen/conda/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/home/hartikainen/.config/fzf/bin \
    PWD=/proc/self/cwd \
    TMPDIR=/tmp \
  /home/hartikainen/.cache/bazel/_bazel_hartikainen/install/81618c1cfcf8a55fe29d247a9003bce4/process-wrapper '--timeout=0' '--kill_delay=15' '--stats=/home/hartikainen/.cache/bazel/_bazel_hartikainen/114ae8b49ccfc70a3ada0f2ba0fbb8a0/sandbox/processwrapper-sandbox/491/stats.out' /usr/lib/llvm-18/bin/clang -U_FORTIFY_SOURCE -fstack-protector -Wall -Wthread-safety -Wself-assign -Wunused-but-set-parameter -Wno-free-nonheap-object -fcolor-diagnostics -fno-omit-frame-pointer '-std=c++14' -MD -MF bazel-out/k8-fastbuild/bin/python/mujoco/_objs/_rollout.so/rollout.pic.d '-frandom-seed=bazel-out/k8-fastbuild/bin/python/mujoco/_objs/_rollout.so/rollout.pic.o' -fPIC -iquote . -iquote bazel-out/k8-fastbuild/bin -iquote external/eigen -iquote bazel-out/k8-fastbuild/bin/external/eigen -iquote external/pybind11 -iquote bazel-out/k8-fastbuild/bin/external/pybind11 -iquote external/python_3_11_x86_64-unknown-linux-gnu -iquote bazel-out/k8-fastbuild/bin/external/python_3_11_x86_64-unknown-linux-gnu -iquote external/abseil-cpp -iquote bazel-out/k8-fastbuild/bin/external/abseil-cpp -isystem include -isystem bazel-out/k8-fastbuild/bin/include -isystem external/eigen -isystem bazel-out/k8-fastbuild/bin/external/eigen -isystem external/pybind11/include -isystem bazel-out/k8-fastbuild/bin/external/pybind11/include -isystem external/python_3_11_x86_64-unknown-linux-gnu/include -isystem bazel-out/k8-fastbuild/bin/external/python_3_11_x86_64-unknown-linux-gnu/include -isystem external/python_3_11_x86_64-unknown-linux-gnu/include/python3.11 -isystem bazel-out/k8-fastbuild/bin/external/python_3_11_x86_64-unknown-linux-gnu/include/python3.11 -isystem external/python_3_11_x86_64-unknown-linux-gnu/include/python3.11m -isystem bazel-out/k8-fastbuild/bin/external/python_3_11_x86_64-unknown-linux-gnu/include/python3.11m -Iexternal/abseil-cpp '-std=c++20' -fexceptions '-fvisibility=hidden' -no-canonical-prefixes -Wno-builtin-macro-redefined '-D__DATE__="redacted"' '-D__TIMESTAMP__="redacted"' '-D__TIME__="redacted"' -c python/mujoco/rollout.cc -o bazel-out/k8-fastbuild/bin/python/mujoco/_objs/_rollout.so/rollout.pic.o)
python/mujoco/rollout.cc:24:10: fatal error: 'threadpool.h' file not found
   24 | #include "threadpool.h"
      |          ^~~~~~~~~~~~~~
1 error generated.
Use --verbose_failures to see the command lines of failed build steps.
INFO: Elapsed time: 19.877s, Critical Path: 11.23s
INFO: 603 processes: 369 internal, 234 processwrapper-sandbox.
ERROR: Build did NOT complete successfully
```
