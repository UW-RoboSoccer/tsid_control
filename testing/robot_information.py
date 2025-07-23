import sys
import pathlib

import mujoco

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from ctrl.conf import RobotConfig

conf = RobotConfig()

# Initialize the MuJoCo model and data
mj_model = mujoco.MjModel.from_xml_path(conf.mjcf)
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = conf.dt

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

print("\n=== MUJOCO ACTUATOR DETAILS ===")
for i in range(mj_model.nu):
    act_name = mj_model.actuator(i).name
    trnid = mj_model.actuator(i).trnid  # [joint_id, qaxis_id]
    print(f"[{i}] actuator_name={act_name}, controls joint={mj_model.joint(trnid[0]).name}")


def get_body_world_positions(mj_model, mj_data):
    body_info = []

    for i in range(mj_model.nbody):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        position = mj_data.xpos[i].copy()  # [x, y, z] world position of body origin
        body_info.append((name, position))

    for name, pos in body_info:
        print(f"{name:20s}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
