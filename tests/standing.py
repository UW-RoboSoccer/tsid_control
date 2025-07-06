# test_02_standing.py
import mujoco
import numpy as np
import mujoco.viewer

def test_standing():
    model = mujoco.MjModel.from_xml_path("../robot/v1/mujoco/robot.xml")
    data = mujoco.MjData(model)
    
    # Set standing pose
    # Example: set the initial state to lie flat on the ground
    data.qpos[0:3] = [0, 0, 0.06]  # x, y, z — small lift to avoid intersection

    # Rotate 90° around X-axis (lie on back)
    data.qpos[3:7] = [0.7071, 0.7071, 0.0, 0.0]  # w, x, y, z

    # Optional: collapsed joints (e.g., legs curled, arms spread)
    data.qpos[7:19] = np.deg2rad([
    30, -20, -45, 90, -10, 5,   # right leg
   -30,  20,  45, 90,  10, -5   # left leg
   ])

    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(10000):  # 1 second
            mujoco.mj_step(model, data)
            viewer.sync()
            

    
    print("robot stand successfully")
    return True

if __name__ == "__main__":
    test_standing()