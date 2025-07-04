import mujoco
import numpy as np
import mujoco.viewer

def test_joint_control():
    model = mujoco.MjModel.from_xml_path("../robot/v1/mujoco/robot.xml")
    data = mujoco.MjData(model)
    
    # Set initial pose
    data.qpos[2] = 0.4
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(2000):  # 2 seconds
            t = i * 0.001
            
            # Simple sinusoidal joint control
            for j in range(min(5, model.nu)):  # First 5 actuators
                data.ctrl[j] = 3 * np.sin(t + j)  # Small movements
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # if data.qpos[2] < 0.1:
            #     print("Robot fell during joint control")
            #     return False
    
    return True

if __name__ == "__main__":
    test_joint_control()