#!/usr/bin/env python3
"""
Simple test script to verify the walking controller works.
This script tests the trajectory generation without running MuJoCo simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import op3_conf as conf
from biped import Biped
from walk_controller import Controller

def test_walking_controller():
    """Test the walking controller trajectory generation."""
    print("Testing walking controller...")
    
    # Initialize the biped robot model
    biped = Biped(conf)
    
    # Initialize the walking controller
    controller = Controller(biped, conf)
    
    # Generate a simple walking path
    print("Generating walking path...")
    v = 0.3  # linear velocity
    w = 0.0  # angular velocity
    T = 5.0  # duration
    dt = conf.dt
    
    # Generate reference trajectory
    t = np.linspace(0, T, int(T / dt))
    traj = np.zeros((len(t), 2))
    x, y = 0.0, 0.0
    for i in range(len(t)):
        x = x + v * dt * np.cos(w * t[i])
        y = y + v * dt * np.sin(w * t[i])
        traj[i, :] = [x, y]
    
    initial_orientation = np.array([1.0, 0.0])  # Forward direction
    
    # Generate footstep trajectory
    print("Generating footsteps...")
    footsteps = controller.gen_footsteps(traj, initial_orientation)
    print(f"Generated {len(footsteps)} footsteps")
    
    # Generate DCM trajectory
    print("Generating DCM trajectory...")
    dcm_endpoints = controller.gen_dcm_traj(depth=3)
    print(f"Generated DCM trajectory with {len(dcm_endpoints)} endpoints")
    
    # Generate ZMP trajectory
    print("Generating ZMP trajectory...")
    controller.gen_zmp_traj()
    print(f"Generated ZMP trajectory with {len(controller.zmp_traj)} steps")
    
    # Plot results
    print("Plotting results...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Trajectory overview
    ax1.set_title('Walking Trajectory Overview')
    
    # Plot reference path
    ax1.plot(traj[:, 0], traj[:, 1], 'k--', alpha=0.5, label='Reference Path')
    
    # Plot footsteps
    for footstep in footsteps:
        foot = footstep[3]
        color = 'red' if foot else 'blue'
        ax1.plot(footstep[0], footstep[1], 'o', color=color, markersize=8)
    
    # Plot DCM endpoints
    for dcm in dcm_endpoints:
        ax1.plot(dcm[0], dcm[1], 'go', markersize=6)
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.axis('equal')
    ax1.grid(True)
    ax1.legend(['Reference Path', 'Right Foot', 'Left Foot', 'DCM Endpoints'])
    
    # Plot 2: Footstep trajectory details
    ax2.set_title('Footstep Trajectory Details')
    
    # Plot footstep trajectories
    for i, footstep in enumerate(controller.footstep_traj):
        foot = footstep[0]
        traj = footstep[1]
        color = 'red' if foot else 'blue'
        for t in traj:
            ax2.plot(t[0, 3], t[1, 3], 'o', color=color, markersize=4, alpha=0.7)
    
    # Plot DCM trajectory
    for dcm in controller.dcm_traj:
        for t in dcm:
            ax2.plot(t[0], t[1], 'go', markersize=3, alpha=0.5)
    
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.axis('equal')
    ax2.grid(True)
    ax2.legend(['Right Foot Traj', 'Left Foot Traj', 'DCM Traj'])
    
    plt.tight_layout()
    plt.show()
    
    print("Test completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_walking_controller()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc() 