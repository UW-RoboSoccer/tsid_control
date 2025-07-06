import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("../../robot/v1/mujoco/robot_actuators_fixed.xml") # update path if needed
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        action = np.random.uniform(-1, 1, model.nu)
        
        ctrl_range = model.actuator_ctrlrange
        ctrl_low = ctrl_range[:, 0]
        ctrl_high = ctrl_range[:, 1]
        scaled_action = ctrl_low + (action + 1.0) * 0.5 * (ctrl_high - ctrl_low)
        data.ctrl[:] = scaled_action

        for _ in range(5):  # sim steps per action
            mujoco.mj_step(model, data)
        
        step += 1
        viewer.sync()
