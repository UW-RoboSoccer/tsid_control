import mujoco.viewer
import mujoco
import numpy as np
import time
import csv

def map_tsid_to_mujoco(q_tsid):
    """Map TSID joint positions to MuJoCo control signals."""
    ctrl = np.zeros(20)  # MuJoCo expects 20 actuators
    
    # Map the 18 actuated joints from TSID to MuJoCo
    ctrl[0] = q_tsid[22]  # right shoulder pitch
    ctrl[1] = q_tsid[23]  # right shoulder roll
    ctrl[2] = q_tsid[24]  # right elbow

    ctrl[3] = q_tsid[14]  # left shoulder pitch
    ctrl[4] = q_tsid[15]  # left shoulder roll
    ctrl[5] = q_tsid[16]  # left elbow

    ctrl[6] = q_tsid[7]   # head yaw
    ctrl[7] = q_tsid[8]   # head pitch

    ctrl[8] = q_tsid[17]  # right hip pitch
    ctrl[9] = q_tsid[18]  # right hip roll
    ctrl[10] = q_tsid[19] # right hip yaw
    ctrl[11] = q_tsid[20] # right knee
    ctrl[12] = q_tsid[21] # right ankle pitch

    ctrl[13] = q_tsid[9]  # left hip pitch
    ctrl[14] = q_tsid[10] # left hip roll
    ctrl[15] = q_tsid[11] # left hip yaw
    ctrl[16] = q_tsid[12] # left knee
    ctrl[17] = q_tsid[13] # left ankle pitch
    
    # The last two actuators (18, 19) are likely ankle roll joints
    ctrl[18] = 0.0  # right ankle roll (if exists)
    ctrl[19] = 0.0  # left ankle roll (if exists)

    return ctrl

def create_stable_standing_config():
    """Create a very stable standing configuration."""
    # This is a 25-element configuration: [base_pos(3), base_quat(4), joints(18)]
    q = np.zeros(25)
    
    # Base position - start higher and let it settle
    q[0] = 0.0  # x
    q[1] = 0.0  # y  
    q[2] = 0.5  # z - start higher
    
    # Base orientation - upright
    q[3] = 1.0  # qw
    q[4] = 0.0  # qx
    q[5] = 0.0  # qy
    q[6] = 0.0  # qz
    
    # Head
    q[7] = 0.0   # head yaw
    q[8] = 0.0   # head pitch
    
    # Left leg - very stable standing pose
    q[9] = 0.0   # left hip pitch
    q[10] = 0.0  # left hip roll  
    q[11] = 0.0  # left hip yaw
    q[12] = 0.1  # left knee - slight bend for stability
    q[13] = -0.1 # left ankle pitch - compensate for knee bend
    
    # Left arm
    q[14] = 0.0  # left shoulder pitch
    q[15] = 0.0  # left shoulder roll
    q[16] = 0.0  # left elbow
    
    # Right leg - very stable standing pose
    q[17] = 0.0  # right hip pitch
    q[18] = 0.0  # right hip roll
    q[19] = 0.0  # right hip yaw
    q[20] = 0.1  # right knee - slight bend for stability
    q[21] = -0.1 # right ankle pitch - compensate for knee bend
    
    # Right arm
    q[22] = 0.0  # right shoulder pitch
    q[23] = 0.0  # right shoulder roll
    q[24] = 0.0  # right elbow
    
    return q

def run_simple_standing():
    """Very simple standing simulation with minimal control."""
    print("Starting simple standing simulation...")
    
    # Load MuJoCo model
    mujoco_model_path = "robot/v1/mujoco/robot.xml"
    mj_model = mujoco.MjModel.from_xml_path(mujoco_model_path)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = 0.002  # 2ms timestep
    
    # Create stable standing configuration
    q = create_stable_standing_config()
    v = np.zeros(mj_model.nv)
    
    print(f"Initial configuration: {q}")
    
    # Set initial state
    mj_data.qpos = q
    mj_data.qvel = v
    
    # Run simulation
    simulation_duration = 10.0  # 10 seconds
    total_steps = int(simulation_duration / mj_model.opt.timestep)
    
    # Open CSV file for logging
    with open('simple_standing_log.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            'step', 'time', 'com_x', 'com_y', 'com_z',
            'left_foot_x', 'left_foot_y', 'left_foot_z',
            'right_foot_x', 'right_foot_y', 'right_foot_z',
            'base_x', 'base_y', 'base_z'
        ])
        
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            print("Starting simple standing simulation...")
            
            for step in range(total_steps):
                if not viewer.is_running():
                    break
                
                # Apply the standing configuration as control
                ctrl = map_tsid_to_mujoco(q)
                mj_data.ctrl = ctrl
                
                # Step simulation
                mujoco.mj_step(mj_model, mj_data)
                
                # Get current state
                com = mj_data.xpos[1]  # CoM position (body 1 is usually the base)
                base_pos = mj_data.qpos[0:3]  # Base position
                
                # Get foot positions (approximate)
                left_foot_pos = mj_data.xpos[2]  # Assuming body 2 is left foot
                right_foot_pos = mj_data.xpos[3]  # Assuming body 3 is right foot
                
                # Log data
                csvwriter.writerow([
                    step, step * mj_model.opt.timestep,
                    com[0], com[1], com[2],
                    left_foot_pos[0], left_foot_pos[1], left_foot_pos[2],
                    right_foot_pos[0], right_foot_pos[1], right_foot_pos[2],
                    base_pos[0], base_pos[1], base_pos[2]
                ])
                
                viewer.sync()
                
                # Print status every 500 steps (1 second)
                if step % 500 == 0:
                    print(f"Step {step}/{total_steps}, Time: {step * mj_model.opt.timestep:.2f}s")
                    print(f"  CoM: {com}")
                    print(f"  Base: {base_pos}")
                    print(f"  Left foot: {left_foot_pos}")
                    print(f"  Right foot: {right_foot_pos}")
                    print("---")
            
            print("Simple standing simulation completed!")
    
    print("Simple standing simulation finished. Check simple_standing_log.csv for data.")

if __name__ == "__main__":
    run_simple_standing() 