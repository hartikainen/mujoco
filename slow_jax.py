import time
import jax
import mujoco
from mujoco import mjx
import numpy as np

# Create a chain of bodies with many geoms and sites
xml = "<mujoco>\n"
xml += "  <worldbody>\n"
xml += '    <body name="0" pos="0 0 0">\n'
xml += '      <joint type="free"/>\n'
xml += '      <geom size="0.1"/>\n'

depth = 9  # > 20 joints
geoms_per_body = 1
sites_per_body = 1

for i in range(1, depth):
    xml += f'      <body name="{i}" pos="0.1 0 0">\n'
    xml += '        <joint type="hinge"/>\n'
    xml += '        <geom size="0.1"/>\n'
    for j in range(geoms_per_body):
        xml += f'        <geom size="0.01" pos="0 {j*0.01} 0"/>\n'
    for j in range(sites_per_body):
        xml += f'        <site name="s_{i}_{j}" pos="0 0 {j*0.01}"/>\n'

for i in range(depth):
    xml += "    </body>"

xml += "\n  </worldbody>\n"
xml += "</mujoco>\n"

m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mx = mjx.put_model(m)
dx = mjx.put_data(m, d)

print(f"Running benchmark with {depth} bodies, {m.ngeom} geoms, {m.nsite} sites on {jax.default_backend()}")

# Compile
kinematics_jit = jax.jit(mjx.kinematics)
dx = kinematics_jit(mx, dx)
dx.qpos.block_until_ready()
print("Compiled.")

# Warmup just in case
for _ in range(2):
    dx = kinematics_jit(mx, dx)
    dx.qpos.block_until_ready()
    
# Benchmark
N = 100
start = time.time()
for _ in range(N):
    dx = kinematics_jit(mx, dx)
    dx.qpos.block_until_ready()
end = time.time()

print(f"Time for {N} iterations: {end - start:.4f}s")
print(f"Avg time: {(end - start)/N*1000:.4f}ms")
