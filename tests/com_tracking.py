import mujoco
import numpy as np
import mujoco.viewer

def test_com_tracking():
    model = mujoco.MjModel.from_xml_path("../robot/v1/mujoco/robot.xml")
    data = mujoco.MjData(model)
    
    data.qpos[2] = 0.4
    com_positions = []
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i in range(1000):
            mujoco.mj_step(model, data)
            
            # Get CoM position (assuming body 1 is torso)
            com = data.xipos[1]
            com_positions.append(com.copy())
            
            viewer.sync()
    
    # Analyze CoM stability
    com_std = np.std(com_positions, axis=0)
    print(f"CoM stability: std_x={com_std[0]:.4f}, std_y={com_std[1]:.4f}, std_z={com_std[2]:.4f}")
    
    if com_std[2] < 0.01:  # CoM height stable within 1cm
        print("✓ CoM tracking stable")
        return True
    else:
        print("✗ CoM unstable")
        return False

if __name__ == "__main__":
    test_com_tracking()