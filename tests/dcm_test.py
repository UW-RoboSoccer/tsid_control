import sys
import numpy as np
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ctrl'))


def test_dcm_planning():
    """Test DCM planning and visualization"""
    from DCM_Planner import DCMPlanner
    from Footstep_Planner import FootstepPlanner, Footstep
    from ZMP_Planner import ZMPPlanner
    
    # Create planners
    footstep_planner = FootstepPlanner(0.1, 0.04)
    dcm_planner = DCMPlanner(np.sqrt(9.81 / 0.4))
    zmp_planner = ZMPPlanner()
    
    # Generate trajectories
    path = [np.array([0, 0]), np.array([0.04, 0]), np.array([0.08, 0])]
    init_supports = [
        Footstep(position=np.array([0, 0.05]), orientation=np.array([0, 0, 0]), side=0),
        Footstep(position=np.array([0, -0.05]), orientation=np.array([0, 0, 0]), side=1)
    ]
    
    footsteps = footstep_planner.plan(path, init_supports)
    dcm_traj = dcm_planner.gen_dcm_traj(footsteps, 1.5)
    zmp_traj = zmp_planner.gen_zmp_traj(footsteps, 1.5)
    
    print(f"Generated {len(dcm_traj)} DCM trajectory steps")
    print("✓ DCM planning works")
    
    # Plot the trajectories
    print("Plotting DCM trajectories...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: DCM trajectory overview
    ax1.set_title('DCM Trajectory Overview')
    
    # Plot footsteps
    for i, footstep in enumerate(footsteps):
        pos = footstep.position[:2]
        side = 'Right' if footstep.side else 'Left'
        color = 'red' if footstep.side else 'blue'
        ax1.plot(pos[0], pos[1], 'o', color=color, markersize=8, label=f'{side} Foot')
    
    # Plot DCM trajectory
    for i, step_dcm in enumerate(dcm_traj):
        dcm_points = np.array(step_dcm)
        ax1.plot(dcm_points[:, 0], dcm_points[:, 1], 'g-', linewidth=2, alpha=0.7, label=f'DCM Step {i+1}')
    
    # Plot ZMP trajectory for reference
    for i, step_zmp in enumerate(zmp_traj):
        zmp_points = np.array(step_zmp)
        ax1.plot(zmp_points[:, 0], zmp_points[:, 1], 'c--', linewidth=1.5, alpha=0.5, label=f'ZMP Step {i+1}')
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.axis('equal')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: DCM vs ZMP comparison
    ax2.set_title('DCM vs ZMP Comparison')
    
    for i, (step_dcm, step_zmp) in enumerate(zip(dcm_traj, zmp_traj)):
        dcm_points = np.array(step_dcm)
        zmp_points = np.array(step_zmp)
        
        ax2.plot(dcm_points[:, 0], dcm_points[:, 1], 'g-', linewidth=2, alpha=0.7, label=f'DCM Step {i+1}')
        ax2.plot(zmp_points[:, 0], zmp_points[:, 1], 'c--', linewidth=2, alpha=0.7, label=f'ZMP Step {i+1}')
    
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.axis('equal')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print("\nTrajectory Statistics:")
    print(f"Number of footsteps: {len(footsteps)}")
    print(f"Number of DCM trajectory steps: {len(dcm_traj)}")
    print(f"Number of ZMP trajectory steps: {len(zmp_traj)}")
    
    # Print first few points of each trajectory
    print("\nFirst few trajectory points:")
    if dcm_traj and dcm_traj[0]:
        print(f"DCM Step 1, Point 1: {dcm_traj[0][0]}")
        print(f"DCM Step 1, Point 5: {dcm_traj[0][4]}")
        print(f"DCM Step 1, Point 10: {dcm_traj[0][9]}")
    if zmp_traj and zmp_traj[0]:
        print(f"ZMP Step 1, Point 1: {zmp_traj[0][0]}")
    
    return True

if __name__ == "__main__":
    test_dcm_planning()