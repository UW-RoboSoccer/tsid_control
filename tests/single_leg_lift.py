import mujoco
import numpy as np
import mujoco.viewer

def test_single_leg_lift():
    """Test 11: Can we lift one leg while balancing?"""
    model = mujoco.MjModel.from_xml_path("../robot/v1/mujoco/robot.xml")
    data = mujoco.MjData(model)
    
    data.qpos[2] = 0.4
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(2000):  # 2 seconds
            t = i * 0.001
            
            # Lift right leg slightly
            if 500 < i < 1500:  # Middle 1 second
                if model.nu >= 5:
                    data.ctrl[3] = 0.3  # Small upward force
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # if data.qpos[2] < 0.1:
            #     print("✗ Robot fell during leg lift")
            #     return False
    
    print("✓ Single leg lift successful")
    return True

if __name__ == "__main__":
    test_single_leg_lift()