import mujoco.viewer
import mujoco
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pinocchio as pin

import op3_conf as conf
from biped import Biped
from walk_controller import Controller

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
    # Set them to 0 for now (they might not be used in this model)
    ctrl[18] = 0.0  # right ankle roll (if exists)
    ctrl[19] = 0.0  # left ankle roll (if exists)

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
    
    # Generate walking path
    print("Generating walking path...")
    traj = generate_walking_path(linear_vel=0.3, angular_vel=0.0, duration=8.0, dt=conf.dt)
    initial_orientation = np.array([1.0, 0.0])  # Forward direction
    
    # Generate footstep trajectory
    footsteps = controller.gen_footsteps(traj, initial_orientation)
    print(f"Generated {len(footsteps)} footsteps")
    
    # Generate DCM trajectory
    dcm_endpoints = controller.gen_dcm_traj(depth=3)
    print(f"Generated DCM trajectory with {len(dcm_endpoints)} endpoints")
    
    # Generate ZMP trajectory
    controller.gen_zmp_traj()
    print("Generated ZMP trajectory")
    
    # Initialize MuJoCo model and data
    print("Loading MuJoCo model...")
    mj_model = mujoco.MjModel.from_xml_path(conf.mujoco_model_path)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = conf.dt
    
    # Initialize simulation state
    q, v = biped.q, biped.v
    mj_data.qpos = q
    
    # Initialize problem data
    i, t = 0, 0.0
    com_0 = biped.robot.com(biped.formulation.data())
    
    # Compute initial problem data
    HQPData = biped.formulation.computeProblemData(t, q, v)
    
    print("Starting MuJoCo simulation...")
    start_time = time.time()
    
    # Main simulation loop
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            t_elapsed = time.time() - start_time
            
            try:
                # Update walking controller
                controller.update()
                
                # Get control references from controller
                com_ref = controller.biped.trajCom.getSample(t).value()
                com_vel_ref = controller.biped.trajCom.getSample(t).derivative()
                com_acc_ref = controller.biped.trajCom.getSample(t).second_derivative()
                
                # Set CoM task references using the correct TSID API
                biped.sample_com.value(com_ref)
                biped.sample_com.derivative(com_vel_ref)
                biped.sample_com.second_derivative(com_acc_ref)
                biped.trajCom.setReference(com_ref)  # Use the vector directly
                
                # Set foot position references using the correct TSID pattern
                if controller.current_step < len(controller.footstep_traj):
                    footstep = controller.footstep_traj[controller.current_step]
                    if footstep[0]:  # Right foot
                        foot_transform = footstep[1][controller.time_step]
                        biped.trajRF.setReference(pin.SE3(foot_transform))
                        biped.rightFootTask.setReference(biped.trajRF.computeNext())
                    else:  # Left foot
                        foot_transform = footstep[1][controller.time_step]
                        biped.trajLF.setReference(pin.SE3(foot_transform))
                        biped.leftFootTask.setReference(biped.trajLF.computeNext())
                
                # Compute control
                biped.comTask.setReference(biped.trajCom.computeNext())
                biped.postureTask.setReference(biped.trajPosture.computeNext())
                biped.rightFootTask.setReference(biped.trajRF.computeNext())
                biped.leftFootTask.setReference(biped.trajLF.computeNext())
                
                # Solve QP problem
                HQPData = biped.formulation.computeProblemData(t, q, v)
                sol = biped.solver.solve(HQPData)
                
                if sol.status != 0:
                    print(f"QP problem could not be solved! Error code: {sol.status}")
                    break
                
                # Get control outputs
                tau = biped.formulation.getActuatorForces(sol)
                dv = biped.formulation.getAccelerations(sol)
                
                # Integrate state
                q, v = biped.integrate_dv(q, v, dv, conf.dt)
                i, t = i + 1, t + conf.dt
                
                # Update MuJoCo
                mj_data.ctrl = map_tsid_to_mujoco(q)
                mujoco.mj_step(mj_model, mj_data)
                
                # Get current state for visualization
                com = biped.robot.com(biped.formulation.data())
                com_vel = biped.robot.com_vel(biped.formulation.data())
                w = np.sqrt(9.81 / com_0[2])
                cp = biped.compute_capture_point(com, com_vel, w)
                
                # Add visual markers
                # CoM reference (red sphere)
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[0],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.05, 0, 0],
                    pos=com_ref,
                    mat=np.eye(3).flatten(),
                    rgba=np.array([1.0, 0.0, 0.0, 1.0]),
                )
                
                # Current CoM (green sphere)
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[1],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.05, 0, 0],
                    pos=com,
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0.0, 1.0, 0.0, 1.0]),
                )
                
                # Left foot (cyan sphere)
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[2],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.025, 0, 0],
                    pos=biped.robot.framePosition(biped.formulation.data(), biped.LF).translation,
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0.0, 1.0, 1.0, 0.5]),
                )
                
                # Right foot (blue sphere)
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[3],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.025, 0, 0],
                    pos=biped.robot.framePosition(biped.formulation.data(), biped.RF).translation,
                    mat=np.eye(3).flatten(),
                    rgba=np.array([0.0, 0.0, 1.0, 0.5]),
                )
                
                # Capture point (yellow cylinder)
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[4],
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[0.025, 0.0001, 0],
                    pos=cp,
                    mat=np.eye(3).flatten(),
                    rgba=np.array([1.0, 1.0, 0.0, 1.0]),
                )
                
                viewer.user_scn.ngeom = 5
                
                # Check if simulation should end
                if controller.current_step >= len(controller.footstep_traj):
                    print("Walking trajectory completed!")
                    break
                
                # Sync viewer
                viewer.sync()
                
            except StopIteration:
                print("Walking trajectory completed!")
                break
            except Exception as e:
                print(f"Error during simulation: {e}")
                break
        
        # Keep viewer open for a moment after simulation ends
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.1)
    
    print("Simulation finished!")
    
    # Plot results
    plot_walking_results(controller, footsteps, dcm_endpoints)

def plot_walking_results(controller, footsteps, dcm_endpoints):
    """Plot the walking results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Trajectory overview
    ax1.set_title('Walking Trajectory Overview')
    
    # Plot footsteps
    for footstep in footsteps:
        foot = footstep[3]
        color = 'red' if foot else 'blue'
        ax1.plot(footstep[0], footstep[1], 'o', color=color, markersize=8)
    
    # Plot DCM endpoints
    for dcm in dcm_endpoints:
        ax1.plot(dcm[0], dcm[1], 'go', markersize=6)
    
    # Plot CoM trajectory
    if controller.logged_com_traj:
        com_traj = np.array(controller.logged_com_traj)
        ax1.plot(com_traj[:, 0], com_traj[:, 1], 'g-', linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.axis('equal')
    ax1.grid(True)
    ax1.legend(['Right Foot', 'Left Foot', 'DCM Endpoints', 'CoM Trajectory'])
    
    # Plot 2: CoM height over time
    ax2.set_title('Center of Mass Height Over Time')
    if controller.logged_com_traj:
        com_traj = np.array(controller.logged_com_traj)
        time_steps = np.arange(len(com_traj)) * conf.dt
        ax2.plot(time_steps, com_traj[:, 2], 'b-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('CoM Height (m)')
        ax2.grid(True)
    
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