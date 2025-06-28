import mujoco.viewer
import mujoco
import numpy as np
import time
import pinocchio as pin
import csv

import op3_conf as conf
from biped import Biped

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

def run_standing_simulation():
    """Simple standing simulation to get the robot stable first."""
    print("Initializing standing simulation...")
    
    # Initialize the biped robot model
    biped = Biped(conf)
    
    # Initialize MuJoCo model and data
    print("Loading MuJoCo model...")
    mj_model = mujoco.MjModel.from_xml_path(conf.mujoco_model_path)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = conf.dt
    
    # Get initial configuration from biped
    q = biped.q0.copy()  # Use the standing configuration
    v = np.zeros(biped.robot.nv)
    
    print(f"Initial q[0:7] (base): {q[0:7]}")
    print(f"Initial q[7:] (joints): {q[7:]}")
    
    # Set MuJoCo initial state
    mj_data.qpos = q
    mj_data.qvel = v
    
    # Get initial foot positions
    biped.formulation.computeProblemData(0.0, q, v)
    left_foot_0 = biped.robot.framePosition(biped.formulation.data(), biped.LF).translation
    right_foot_0 = biped.robot.framePosition(biped.formulation.data(), biped.RF).translation
    com_0 = biped.robot.com(biped.formulation.data())
    
    print(f"Initial CoM: {com_0}")
    print(f"Left foot: {left_foot_0}")
    print(f"Right foot: {right_foot_0}")
    
    # Check if feet are on the ground
    if left_foot_0[2] > 0.01 or right_foot_0[2] > 0.01:
        print("WARNING: Feet are not on the ground! Adjusting height...")
        # Lower the robot until feet touch the ground
        while left_foot_0[2] > 0.01 or right_foot_0[2] > 0.01:
            q[2] -= 0.01  # Lower the robot
            biped.formulation.computeProblemData(0.0, q, v)
            left_foot_0 = biped.robot.framePosition(biped.formulation.data(), biped.LF).translation
            right_foot_0 = biped.robot.framePosition(biped.formulation.data(), biped.RF).translation
            print(f"Lowering robot, feet at: left={left_foot_0[2]:.3f}, right={right_foot_0[2]:.3f}")
            if q[2] < 0.1:  # Safety limit
                break
        
        mj_data.qpos = q
        com_0 = biped.robot.com(biped.formulation.data())
        print(f"Adjusted CoM: {com_0}")
    
    # Initialize simulation variables
    i, t = 0, 0.0
    start_time = time.time()
    simulation_duration = 5.0  # 5 seconds of standing
    total_steps = int(simulation_duration / conf.dt)
    
    # Open CSV file for logging
    with open('standing_debug_log.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            'step', 'sim_time', 'com_x', 'com_y', 'com_z',
            'left_foot_x', 'left_foot_y', 'left_foot_z',
            'right_foot_x', 'right_foot_y', 'right_foot_z',
            'qp_status', 'com_error'
        ])
        
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            print("Starting standing simulation...")
            
            for step in range(total_steps):
                if not viewer.is_running():
                    break
                
                t_elapsed = time.time() - start_time
                
                # Keep CoM at initial position during standing
                com_ref = com_0.copy()
                biped.sample_com.value(com_ref)
                biped.sample_com.derivative(np.zeros(3))
                biped.sample_com.second_derivative(np.zeros(3))
                biped.trajCom.setReference(com_ref)
                
                # Set all task references
                biped.comTask.setReference(biped.trajCom.computeNext())
                biped.postureTask.setReference(biped.trajPosture.computeNext())
                biped.rightFootTask.setReference(biped.trajRF.computeNext())
                biped.leftFootTask.setReference(biped.trajLF.computeNext())
                
                # Solve QP problem
                HQPData = biped.formulation.computeProblemData(t, q, v)
                sol = biped.solver.solve(HQPData)
                
                if sol.status != 0:
                    print(f"QP problem could not be solved! Error code: {sol.status}")
                    print("Stopping simulation due to QP failure.")
                    break
                
                dv = biped.formulation.getAccelerations(sol)
                q, v = biped.integrate_dv(q, v, dv, conf.dt)
                i, t = i + 1, t + conf.dt
                
                # Apply control to MuJoCo
                mj_data.ctrl = map_tsid_to_mujoco(q)
                mujoco.mj_step(mj_model, mj_data)
                
                # Get current state
                biped.formulation.computeProblemData(t, q, v)
                com_true = biped.robot.com(biped.formulation.data())
                left_foot_pos = biped.robot.framePosition(biped.formulation.data(), biped.LF).translation
                right_foot_pos = biped.robot.framePosition(biped.formulation.data(), biped.RF).translation
                
                # Calculate CoM error
                com_error = np.linalg.norm(com_true - com_ref)
                
                # Log data
                csvwriter.writerow([
                    i, t_elapsed,
                    com_true[0], com_true[1], com_true[2],
                    left_foot_pos[0], left_foot_pos[1], left_foot_pos[2],
                    right_foot_pos[0], right_foot_pos[1], right_foot_pos[2],
                    sol.status, com_error
                ])
                
                viewer.sync()
                
                # Print status every 500 steps (1 second)
                if step % 500 == 0:
                    print(f"Step {step}/{total_steps}, Time: {t:.2f}s")
                    print(f"  CoM: {com_true}")
                    print(f"  CoM error: {com_error:.4f}")
                    print(f"  Left foot: {left_foot_pos}")
                    print(f"  Right foot: {right_foot_pos}")
                    print(f"  QP status: {sol.status}")
                    print("---")
            
            print("Standing simulation completed!")
    
    print("Standing simulation finished. Check standing_debug_log.csv for data.")

if __name__ == "__main__":
    run_standing_simulation() 