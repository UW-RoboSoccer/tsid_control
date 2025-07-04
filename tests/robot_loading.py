import mujoco
import numpy as np

def test_robot_loading():
    try:
        model = mujoco.MjModel.from_xml_path("../robot/v1/mujoco/robot.xml")
        data = mujoco.MjData(model)
        
        print("robot loaded successfully")
        print(f"  - DOF: {model.nq}")
        print(f"  - Actuators: {model.nu}")
        print(f"  - Bodies: {model.nbody}")
        
        return True
    except Exception as e:
        print(f"robot loading failed: {e}")
        return False

if __name__ == "__main__":
    test_robot_loading()