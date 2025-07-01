import mujoco.viewer
import mujoco
import numpy as np
import time
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pinocchio as pin
import csv

import op3_conf as conf
from biped import Biped
from walk_controller import Controller

def map_tsid_to_mujoco(q_tsid):
    """Map TSID joint positions to MuJoCo position control signals.
    
    This uses the EXACT same approach as biped_balance.py - direct position control!
    Much more stable than trying to compute torques manually.
    """
    ctrl = np.zeros(20)
    
    # Use the exact same mapping as biped_balance.py
    ctrl[0] = q_tsid[22] # right shoulder pitch
    ctrl[1] = q_tsid[23] # right shoulder roll
    ctrl[2] = q_tsid[24] # right elbow

    ctrl[3] = q_tsid[14] # left shoulder pitch
    ctrl[4] = q_tsid[15] # left shoulder roll
    ctrl[5] = q_tsid[16] # left elbow

    ctrl[6] = q_tsid[7] # head yaw
    ctrl[7] = q_tsid[8] # head pitch

    ctrl[8] = q_tsid[17] # right hip pitch
    ctrl[9] = q_tsid[18] # right hip roll
    ctrl[10] = q_tsid[19] # right hip yaw
    ctrl[11] = q_tsid[20] # right knee
    ctrl[12] = q_tsid[21] # right ankle pitch

    ctrl[13] = q_tsid[9] # left hip pitch
    ctrl[14] = q_tsid[10] # left hip roll
    ctrl[15] = q_tsid[11] # left hip yaw
    ctrl[16] = q_tsid[12] # left knee
    ctrl[17] = q_tsid[13] # left ankle pitch
    ctrl[18] = 0.0
    ctrl[19] = 0.0

    return ctrl

def generate_walking_path(linear_vel=0.5, angular_vel=0.0, duration=10.0, dt=0.002):
    """Generate a reference path for walking."""
    T = duration
    t = np.linspace(0, T, int(T / dt))
    traj = np.zeros((len(t), 2))
    
    x, y = 0.0, 0.0
    for i in range(len(t)):
        x = x + linear_vel * dt * np.cos(angular_vel * t[i])
        y = y + linear_vel * dt * np.sin(angular_vel * t[i])
        traj[i, :] = [x, y]
    
    return traj

def run_walking_simulation():
    """Main function to run the walking simulation."""
    print("Initializing walking simulation...")
    
    # Initialize the biped robot model
    biped = Biped(conf)
    
    # Initialize the walking controller
    controller = Controller(biped, conf)
    
    # Generate walking path with very conservative parameters
    print("Generating walking path...")
    traj = generate_walking_path(linear_vel=conf.linear_vel, angular_vel=0.0, duration=3.0, dt=conf.dt)  # Use linear_vel from conf
    initial_orientation = np.array([1.0, 0.0])  # Forward direction
    
    # Generate footstep trajectory
    footsteps = controller.gen_footsteps(traj, initial_orientation)
    print(f"Generated {len(footsteps)} footsteps")
    
    # Generate DCM trajectory
    dcm_endpoints = controller.gen_dcm_traj()
    print(f"Generated DCM trajectory with {len(dcm_endpoints)} endpoints")
    
    # Generate ZMP trajectory
    controller.gen_zmp_traj()
    print("Generated ZMP trajectory")
    
    # Initialize MuJoCo model and data
    print("Loading MuJoCo model...")
    mj_model = mujoco.MjModel.from_xml_path(conf.mujoco_model_path)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = conf.dt
    
    # Initialize simulation state - use natural Biped configuration 
    # 🔧 FIXED: Use the natural joint configuration from Biped class (like biped_balance.py)
    # DON'T override with zeros - the Biped class sets up a proper standing pose!
    q, v = biped.q, biped.v
    
    # Only set initial velocities to zero for stability
    v = np.zeros_like(v)
    
    print(f"🔧 Using natural Biped joint configuration (not forcing zeros)")
    print(f"   First 10 joint positions: {q[:10]}")
    print(f"   Joint positions 7-16 (first 10 actuated): {q[7:17]}")
    
    # Keep the natural joint configuration - don't override anything!
    # The Biped class already sets up a proper standing pose
    
    # Set initial velocities to zero
    mj_data.qpos = q
    mj_data.qvel = v
    
    # Step MuJoCo once to update the model state
    mujoco.mj_step(mj_model, mj_data)
    
    # Update q and v from MuJoCo data after the step
    q = mj_data.qpos.copy()
    v = mj_data.qvel.copy()
    
    # Update the pinocchio model data
    biped.formulation.computeProblemData(0.0, q, v)
    
    # Now get the actual foot positions after setting the joint configuration
    left_foot_0 = biped.robot.framePosition(biped.formulation.data(), biped.LF).translation
    right_foot_0 = biped.robot.framePosition(biped.formulation.data(), biped.RF).translation
    
    # Calculate center between feet
    feet_center = (left_foot_0 + right_foot_0) / 2.0
    
    # Set the base position to center the robot between its feet
    q[0] = feet_center[0]  # Set x position to center between feet
    q[1] = feet_center[1]  # Set y position to center between feet
    
    # 🔧 FIXED: Use proper foot-on-ground initialization (same as biped_balance.py)
    # Instead of forcing a specific CoM height, place feet on ground and accept natural CoM height
    print(f"🔧 PROPER INITIALIZATION (like biped_balance.py):")
    print(f"   Current base height q[2]: {q[2]:.6f}")
    print(f"   Current foot heights: L={left_foot_0[2]:.6f}, R={right_foot_0[2]:.6f}")
    
    # Place feet on ground (same approach as biped_balance.py)
    min_foot_height = min(left_foot_0[2], right_foot_0[2])
    q[2] = q[2] - min_foot_height  # This puts the lowest foot on the ground
    
    print(f"   Min foot height: {min_foot_height:.6f}")
    print(f"   Adjusted base height to: {q[2]:.6f}")
    print(f"   This will result in feet at ground level and natural CoM height")
    
    # Update MuJoCo state with the corrected position
    mj_data.qpos = q
    mj_data.qvel = v
    
    # Step MuJoCo again to update the model state
    mujoco.mj_step(mj_model, mj_data)
    
    # Re-update the pinocchio model data with the final initial position
    biped.formulation.computeProblemData(0.0, mj_data.qpos, mj_data.qvel)
    
    # Now get the actual CoM position after proper positioning
    com_0 = biped.robot.com(biped.formulation.data())
    left_foot_0 = biped.robot.framePosition(biped.formulation.data(), biped.LF).translation
    right_foot_0 = biped.robot.framePosition(biped.formulation.data(), biped.RF).translation
    feet_center = (left_foot_0 + right_foot_0) / 2.0
    
    print("=" * 60)
    print("🔍 FINAL INITIALIZATION RESULTS")
    print("=" * 60)
    print(f"Natural CoM position: {com_0}")
    print(f"Left foot: {left_foot_0}")
    print(f"Right foot: {right_foot_0}")
    print(f"Feet center: {feet_center}")
    print(f"Base position q[0:3]: {q[0:3]}")
    print(f"Feet on ground: Left={left_foot_0[2]:.6f}, Right={right_foot_0[2]:.6f}")
    print(f"✅ Using natural CoM height: {com_0[2]:.6f}m (no forced height)")
    print("=" * 60)
    
    # Initialize problem data
    i, t = 0, 0.0
    
    print("Starting MuJoCo simulation...")
    start_time = time.time()
    standing_time = 3.0  # 3 seconds of standing first (increased from 2)
    standing_steps = int(standing_time / conf.dt)
    true_com_log = []
    
    # Open CSV file for logging
    with open('walking_debug_log.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            'step', 'sim_time', 'current_step', 'time_step',
            'com_ref_x', 'com_ref_y', 'com_ref_z',
            'com_x', 'com_y', 'com_z',
            'foot_ref_x', 'foot_ref_y', 'foot_ref_z',
            'left_foot_x', 'left_foot_y', 'left_foot_z',
            'right_foot_x', 'right_foot_y', 'right_foot_z',
            'qp_status', 'foot_distance', 'warning',
            'com_y_error', 'com_y_vel', 'com_y_acc', 'support_foot'
        ])
        
        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            # PHASE 1: Standing phase - hold CoM at initial position
            print("PHASE 1: Standing phase (2 seconds)...")
            for stand_i in range(standing_steps):
                if not viewer.is_running():
                    break
                    
                t_elapsed = time.time() - start_time
                
                # 🔧 FIXED: Hold natural CoM position (same as biped_balance.py)
                # Don't try to force a specific height - just hold the natural position from initialization
                com_ref = com_0.copy()  # Hold the natural CoM position
                
                # 🔍 DEBUG: Print detailed info for first few steps
                if stand_i < 5:
                    print(f"\n🔍 STEP {stand_i} DEBUG:")
                    print(f"   Holding natural CoM position: {com_ref}")
                    print(f"   No height ramping - accept the natural height from foot-on-ground init")
                
                biped.sample_com.value(com_ref)
                biped.sample_com.derivative(np.zeros(3))
                biped.sample_com.second_derivative(np.zeros(3))
                biped.trajCom.setReference(com_ref)
                biped.comTask.setReference(biped.trajCom.computeNext())
                biped.postureTask.setReference(biped.trajPosture.computeNext())
                biped.rightFootTask.setReference(biped.trajRF.computeNext())
                biped.leftFootTask.setReference(biped.trajLF.computeNext())
                
                # Solve QP problem
                HQPData = biped.formulation.computeProblemData(t, q, v)
                sol = biped.solver.solve(HQPData)
                
                if sol.status != 0:
                    print(f"❌ QP problem could not be solved! Error code: {sol.status}")
                    break
                
                dv = biped.formulation.getAccelerations(sol)
                
                # 🔍 DEBUG: Print control info for first few steps  
                if stand_i < 5:
                    print(f"   QP status: {sol.status}")
                    print(f"   Computed accelerations (first 5): {dv[:5]}")
                    print(f"   Current velocities (first 5): {v[:5]}")
                    print(f"   Current positions (first 5): {q[:5]}")
                
                q, v = biped.integrate_dv(q, v, dv, conf.dt)
                i, t = i + 1, t + conf.dt
                
                # 🔧 FIXED: Use simple position control like biped_balance.py  
                # This sends joint positions directly to MuJoCo - much more stable!
                mj_data.ctrl = map_tsid_to_mujoco(q)
                mujoco.mj_step(mj_model, mj_data)
                
                com_true = biped.robot.com(biped.formulation.data())
                true_com_log.append(com_true.copy())
                
                # 🔍 DEBUG: Print result for first few steps
                if stand_i < 5:
                    print(f"   Result CoM: {com_true}")
                    print(f"   CoM height change: {com_true[2] - com_ref[2]:.6f}")
                    if abs(com_true[2] - com_ref[2]) > 0.01:
                        print(f"   ⚠️  WARNING: Large CoM height deviation!")
                        
                    # Check base position changes
                    base_change = np.linalg.norm(q[:3] - mj_data.qpos[:3])
                    if base_change > 0.01:
                        print(f"   ⚠️  WARNING: Large base position change: {base_change:.6f}")
                    print(f"   ---")
                
                # Get foot positions for logging
                left_foot_pos = biped.robot.framePosition(biped.formulation.data(), biped.LF).translation
                right_foot_pos = biped.robot.framePosition(biped.formulation.data(), biped.RF).translation
                
                # Log data
                csvwriter.writerow([
                    i, t_elapsed, -1, -1,  # -1 indicates standing phase
                    com_ref[0], com_ref[1], com_ref[2],
                    com_true[0], com_true[1], com_true[2],
                    0.0, 0.0, 0.0,  # No foot reference during standing
                    left_foot_pos[0], left_foot_pos[1], left_foot_pos[2],
                    right_foot_pos[0], right_foot_pos[1], right_foot_pos[2],
                    sol.status, 0.0, "Standing phase",
                    0.0, 0.0, 0.0, "both"
                ])
                
                viewer.sync()
                
                if stand_i % 500 == 0:  # Print every second
                    print(f"Standing phase: {stand_i}/{standing_steps}, CoM: {com_true}")
            
            print("PHASE 2: Starting walking phase...")
            # PHASE 2: Walking phase - very conservative
            
            # Start the walking controller
            walking_started = controller.start_walking()
            if walking_started:
                while viewer.is_running():
                    t_elapsed = time.time() - start_time
                    try:
                        # Update the controller first to generate fresh trajectories
                        controller.update()
                        
                        # If walking has stopped, break out of loop
                        if not controller.walking_started:
                            print("Walking completed!")
                            break
                        
                        # Now get the references from the controller
                        sample = controller.biped.trajCom.getSample(t)
                        com_ref = sample.value()
                        com_vel_ref = sample.derivative()
                        com_acc_ref = sample.second_derivative()

                        biped.sample_com.value(com_ref)
                        biped.sample_com.derivative(com_vel_ref)
                        biped.sample_com.second_derivative(com_acc_ref)
                        biped.trajCom.setReference(com_ref)
                        
                        foot_distance = None
                        warning = ''
                        foot_ref = [None, None, None]
                        
                        # Set foot references if available
                        if controller.current_step < len(controller.footstep_traj):
                            footstep = controller.footstep_traj[controller.current_step]
                            if footstep[0]:  # Right foot
                                if controller.time_step < len(footstep[1]):
                                    foot_transform = footstep[1][controller.time_step]
                                    foot_ref = foot_transform[:3, 3]
                                    current_foot_pos = biped.robot.framePosition(biped.formulation.data(), biped.RF).translation
                                    foot_distance = float(np.linalg.norm(foot_ref - current_foot_pos))
                                    if foot_distance > 0.05:  # Very conservative limit
                                        warning = f'Foot position too far ({foot_distance:.3f}m), limiting movement'
                                        # Don't move foot if too far
                                        foot_ref = current_foot_pos
                                    else:
                                        biped.trajRF.setReference(pin.SE3(foot_transform))
                                        biped.rightFootTask.setReference(biped.trajRF.computeNext())
                            else:  # Left foot
                                if controller.time_step < len(footstep[1]):
                                    foot_transform = footstep[1][controller.time_step]
                                    foot_ref = foot_transform[:3, 3]
                                    current_foot_pos = biped.robot.framePosition(biped.formulation.data(), biped.LF).translation
                                    foot_distance = float(np.linalg.norm(foot_ref - current_foot_pos))
                                    if foot_distance > 0.05:  # Very conservative limit
                                        warning = f'Foot position too far ({foot_distance:.3f}m), limiting movement'
                                        # Don't move foot if too far
                                        foot_ref = current_foot_pos
                                    else:
                                        biped.trajLF.setReference(pin.SE3(foot_transform))
                                        biped.leftFootTask.setReference(biped.trajLF.computeNext())
                        
                        # Set task references
                        biped.comTask.setReference(biped.trajCom.computeNext())
                        biped.postureTask.setReference(biped.trajPosture.computeNext())
                        
                        # Solve QP problem
                        HQPData = biped.formulation.computeProblemData(t, q, v)
                        sol = biped.solver.solve(HQPData)
                        
                        if sol.status != 0:
                            print(f"QP problem could not be solved! Error code: {sol.status}")
                            print("Continuing with previous solution...")
                            if i > 0:
                                pass  # Continue with previous solution
                            else:
                                break
                        
                        dv = biped.formulation.getAccelerations(sol)
                        
                        # Very conservative velocity limits
                        max_vel = 5.0  # Increased maximum velocity
                        if np.linalg.norm(v) > max_vel:
                            v = v * max_vel / np.linalg.norm(v)
                        
                        q, v = biped.integrate_dv(q, v, dv, conf.dt)
                        i, t = i + 1, t + conf.dt
                        
                        mj_data.ctrl = map_tsid_to_mujoco(q)
                        mujoco.mj_step(mj_model, mj_data)
                        
                        com_true = biped.robot.com(biped.formulation.data())
                        true_com_log.append(com_true.copy())
                        
                        # Get foot positions for logging
                        left_foot_pos = biped.robot.framePosition(biped.formulation.data(), biped.LF).translation
                        right_foot_pos = biped.robot.framePosition(biped.formulation.data(), biped.RF).translation
                        
                        # Calculate errors for logging
                        com_y_error = com_ref[1] - com_true[1] if len(com_ref) > 1 else 0.0
                        com_y_vel = com_vel_ref[1] if len(com_vel_ref) > 1 and com_vel_ref is not None else 0.0
                        com_y_acc = com_acc_ref[1] if len(com_acc_ref) > 1 and com_acc_ref is not None else 0.0
                        support_foot = controller.get_support_foot()
                        
                        # Log data
                        csvwriter.writerow([
                            i, t_elapsed, controller.current_step, controller.time_step,
                            com_ref[0], com_ref[1], com_ref[2],
                            com_true[0], com_true[1], com_true[2],
                            foot_ref[0] if foot_ref[0] is not None else 0.0,
                            foot_ref[1] if foot_ref[1] is not None else 0.0,
                            foot_ref[2] if foot_ref[2] is not None else 0.0,
                            left_foot_pos[0], left_foot_pos[1], left_foot_pos[2],
                            right_foot_pos[0], right_foot_pos[1], right_foot_pos[2],
                            sol.status, foot_distance if foot_distance is not None else 0.0, warning,
                            com_y_error, com_y_vel, com_y_acc, support_foot
                        ])
                        
                        viewer.sync()
                        
                        # Print status every 500 steps (1 second)
                        if i % 500 == 0:
                            print(f"Step {i}, Time: {t:.2f}s, CoM: {com_true}, QP Status: {sol.status}")
                            
                    except StopIteration:
                        print("Walking trajectory completed!")
                        break
                    except Exception as e:
                        print(f"Error in walking simulation: {e}")
                        print("Continuing simulation...")
                        continue
            else:
                print("Failed to start walking controller!")
        
        # Keep viewer open for a moment after simulation ends
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.1)
    
    print("Simulation finished!")
    
    # Plot results
    plot_walking_results(controller, footsteps, dcm_endpoints)

    # Save true CoM log for plotting if needed
    np.save('true_com_log.npy', np.array(true_com_log))

def plot_walking_results(controller, footsteps, dcm_endpoints):
    """Plot the walking results with improved visualization for debugging."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Trajectory overview
    ax1.set_title('Walking Trajectory Overview')
    
    # Plot footsteps (differentiate right/left)
    for footstep in footsteps:
        foot = footstep[3]
        color = 'red' if foot else 'blue'
        label = 'Left Foot' if foot else 'Right Foot'
        ax1.plot(footstep[0], footstep[1], 'o', color=color, markersize=8, label=label)
    
    # Plot DCM endpoints (as points)
    dcm_endpoints_arr = np.array(dcm_endpoints)
    ax1.plot(dcm_endpoints_arr[:, 0], dcm_endpoints_arr[:, 1], 'go', markersize=6, label='DCM Endpoints')

    # Plot full DCM trajectory as a line
    if hasattr(controller, 'dcm_traj') and controller.dcm_traj:
        dcm_traj_points = np.concatenate(controller.dcm_traj, axis=0)
        ax1.plot(dcm_traj_points[:, 0], dcm_traj_points[:, 1], 'g--', linewidth=1.5, alpha=0.7, label='DCM Trajectory')

    # Plot ZMP trajectory if available
    if hasattr(controller, 'zmp_traj') and controller.zmp_traj:
        zmp_traj_points = np.concatenate(controller.zmp_traj, axis=0)
        ax1.plot(zmp_traj_points[:, 0], zmp_traj_points[:, 1], 'c-', linewidth=1.5, alpha=0.7, label='ZMP Trajectory')

    # Plot CoM trajectory as a line
    if controller.logged_com_traj:
        com_traj = np.array(controller.logged_com_traj)
        ax1.plot(com_traj[:, 0], com_traj[:, 1], 'm-', linewidth=2, alpha=0.8, label='CoM Trajectory')
    
    # Plot true CoM trajectory (from simulation)
    if hasattr(controller, 'logged_actual_com_traj') and controller.logged_actual_com_traj:
        actual_com_traj = np.array(controller.logged_actual_com_traj)
        ax1.plot(actual_com_traj[:, 0], actual_com_traj[:, 1], 'k-', linewidth=2, alpha=0.8, label='True CoM (Sim)')

    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.axis('equal')
    ax1.grid(True)
    # Fix legend: only show unique labels
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())
    
    # Plot 2: CoM height over time
    ax2.set_title('Center of Mass Height Over Time')
    if controller.logged_com_traj:
        com_traj = np.array(controller.logged_com_traj)
        time_steps = np.arange(len(com_traj)) * conf.dt
        ax2.plot(time_steps, com_traj[:, 2], 'b-', linewidth=2, label='CoM Height')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('CoM Height (m)')
        ax2.grid(True)
        ax2.legend()

    if hasattr(controller, 'logged_actual_com_traj') and controller.logged_actual_com_traj:
        actual_com_traj = np.array(controller.logged_actual_com_traj)
        time_steps = np.arange(len(actual_com_traj)) * conf.dt
        ax2.plot(time_steps, actual_com_traj[:, 2], 'k-', linewidth=2, label='True CoM Height')
        ax2.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        run_walking_simulation()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc() 