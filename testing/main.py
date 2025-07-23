import time
import sys
import pathlib

import mujoco
import mujoco.viewer

sys.path.append(str(pathlib.Path(__file__).parent.parent))
from testing.test_controller import test_controller
from testing.robot_information import mj_model, mj_data, conf, get_body_world_positions

start_time = time.time()

num_cycles = 0

test_controller.set_initial_pose(mj_model, mj_data)
mujoco.mj_forward(mj_model, mj_data)
print(f"qpos before forward: {mj_data.qpos}")
test_controller.align_on_ground(mj_model, mj_data)
mujoco.mj_forward(mj_model, mj_data)
print(f"qpos after align: {mj_data.qpos}")
get_body_world_positions(mj_model, mj_data)


# Main simulation loop
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        if num_cycles % 1000 == 0:
            print(f"qpos: {mj_data.qpos}")
            get_body_world_positions(mj_model, mj_data)

        num_cycles += 1

        t_elapsed = time.time() - start_time

        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD

        viewer.user_scn.ngeom = 4

        mujoco.mj_step(mj_model, mj_data)

        viewer.sync()

        time.sleep(conf.dt / 2)

print("Simulation finished!")
