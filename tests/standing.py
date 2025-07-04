# test_02_standing.py
import mujoco
import numpy as np
import mujoco.viewer

def test_standing():
    model = mujoco.MjModel.from_xml_path("../robot/v1/mujoco/robot.xml")
    data = mujoco.MjData(model)
    
    # Set standing pose
    data.qpos[2] = 0.4  # Height
    data.qpos[7:17] = 0.0  # Joint positions (adjust indices as needed)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(10000):  # 1 second
            mujoco.mj_step(model, data)
            viewer.sync()
            
            if data.qpos[2] < 0.1:  # Fell below 10cm
                print("Robot fell! fuck")
                return False
    
    print("robot stand successfully")
    return True

if __name__ == "__main__":
    test_standing()