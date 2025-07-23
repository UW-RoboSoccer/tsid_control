import time
import sys
import pathlib

import mujoco
import mujoco.viewer

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from ctrl.conf import RobotConfig

conf = RobotConfig()

# Initialize the MuJoCo model and data
mj_model = mujoco.MjModel.from_xml_path(conf.mjcf)
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = conf.dt

start_time = time.time()

# Print joint structure for both Pinocchio and MuJoCo
print("Size of mujoco qpos: ", mj_data.ctrl.shape)

print("\n=== MUJOCO JOINT STRUCTURE ===")
print(f"nq: {mj_model.nq}, nu: {mj_model.nu}")
print("Joint ordering in Mujoco:")
for i in range(mj_model.njnt):
    joint_name = mj_model.joint(i).name
    joint_type = mj_model.joint(i).type
    joint_qpos_addr = mj_model.joint(i).qposadr
    joint_actuator_idx = -1
    # Find corresponding actuator if any
    for j in range(mj_model.nu):
        if mj_model.actuator(j).trnid[0] == i:
            joint_actuator_idx = j
            break
    print(
        f"Joint {i}: {joint_name}, type: {joint_type}, qpos_addr: {joint_qpos_addr}, actuator_idx: {joint_actuator_idx}"
    )

# Main simulation loop
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        t_elapsed = time.time() - start_time

        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD

        viewer.user_scn.ngeom = 4

        mujoco.mj_step(mj_model, mj_data)

        viewer.sync()

        time.sleep(conf.dt / 2)

print("Simulation finished!")
