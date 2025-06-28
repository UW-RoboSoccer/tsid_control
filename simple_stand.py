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
    ctrl[13] = q_tsid[25] # right ankle roll

    ctrl[14] = q_tsid[9]  # left hip pitch
    ctrl[15] = q_tsid[10] # left hip roll
    ctrl[16] = q_tsid[11] # left hip yaw
    ctrl[17] = q_tsid[12] # left knee
    ctrl[18] = q_tsid[13] # left ankle pitch
    ctrl[19] = q_tsid[26] # left ankle roll
    
    return ctrl

def main():
    # Load the model
    model = mujoco.MjModel.from_xml_path("robot/v1/mujoco/robot.xml")
    data = mujoco.MjData(model)
    
    # Initial standing configuration (27 DOFs: 6 base + 20 actuated + 1 head)
    q_init = np.zeros(27)
    
    # Set the robot to start above the ground
    # Base position: x=0, y=0, z=0.4 (40cm above ground)
    # Base orientation: roll=0, pitch=0, yaw=0
    q_init[2] = 0.4  # Set Z position to 40cm above ground
    
    # Set initial configuration
    data.qpos[:] = q_init
    mujoco.mj_forward(model, data)
    
    # Simple standing controller - just hold the initial position
    target_positions = q_init.copy()
    
    # Controller gains (extremely conservative to prevent instability)
    kp = 1.0  # position gain (very low)
    kd = 0.1  # velocity gain (very low)
    
    # Open CSV file for logging
    with open('simple_stand_log.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time', 'com_x', 'com_y', 'com_z', 'com_vx', 'com_vy', 'com_vz'])
        
        # Simulation loop
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()
            
            while viewer.is_running():
                step_start = time.time()
                
                # Get current joint positions and velocities
                current_pos = data.qpos.copy()  # 27 elements
                current_vel = data.qvel.copy()  # 26 elements (no velocity for free joint)
                
                # Simple PD controller to hold position (ONLY for actuated joints)
                # Don't try to control the base (free joint) - let it move freely
                
                # Target positions for actuated joints only (indices 6-25)
                target_actuated = target_positions[6:26]
                current_actuated = current_pos[6:26]
                current_actuated_vel = current_vel[6:26]  # Velocities for actuated joints
                
                # Compute control torques (only for actuated joints)
                actuated_pos_error = target_actuated - current_actuated
                actuated_vel_error = -current_actuated_vel  # target velocity is 0
                tau_actuated = kp * actuated_pos_error + kd * actuated_vel_error
                
                # Apply control (only to actuated joints)
                data.ctrl[:] = tau_actuated
                
                # Step simulation
                mujoco.mj_step(model, data)
                
                # Log data
                com_pos = data.xpos[1]  # CoM position
                com_vel = data.qvel[:3]  # CoM velocity
                current_time = time.time() - start_time
                
                writer.writerow([
                    current_time,
                    com_pos[0], com_pos[1], com_pos[2],
                    com_vel[0], com_vel[1], com_vel[2]
                ])
                
                # Sync with real time
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                
                # Stop after 5 seconds
                if current_time > 5.0:
                    break

if __name__ == "__main__":
    main() 