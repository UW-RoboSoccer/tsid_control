# test_08_complete_library.py
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ctrl'))
import numpy as np

def test_complete_library():
    from const import dt, step_height, step_width, step_length, step_duration
    from Footstep_Planner import FootstepPlanner, Footstep
    from DCM_Planner import DCMPlanner
    from CoMPlanner import CoMPlanner
    from ZMP_Planner import ZMPPlanner
    
    # Create all planners
    footstep_planner = FootstepPlanner(step_width, step_length)
    dcm_planner = DCMPlanner(np.sqrt(9.81 / 0.4))
    com_planner = CoMPlanner(np.sqrt(9.81 / 0.4))
    zmp_planner = ZMPPlanner()
    
    # Generate complete walking scenario
    path = [np.array([0, 0]), np.array([step_length, 0]), np.array([2*step_length, 0])]
    init_supports = [
        Footstep(position=np.array([0, step_width/2]), orientation=np.array([0, 0, 0]), side=0),
        Footstep(position=np.array([0, -step_width/2]), orientation=np.array([0, 0, 0]), side=1)
    ]
    
    footsteps = footstep_planner.plan(path, init_supports)
    dcm_traj = dcm_planner.gen_dcm_traj(footsteps, step_duration)
    zmp_traj = zmp_planner.gen_zmp_traj(footsteps, step_duration)
    com_traj = com_planner.gen_com_traj(dcm_traj, np.array([0, 0]), np.array([0, 0]))
    
    print(f"Generated {len(footsteps)} footsteps")
    print(f"Generated {len(dcm_traj)} DCM trajectory steps")
    print(f"Generated {len(zmp_traj)} ZMP trajectory steps")
    print(f"Generated {len(com_traj)} CoM trajectory steps")
    
    # Create comprehensive visualization
    print("\nCreating comprehensive visualization...")
    fig = plt.figure(figsize=(20, 12))
    
    # Create a 2x3 subplot layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Complete Overview (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Complete Walking Overview', fontsize=12, fontweight='bold')
    
    # Plot footsteps
    for i, footstep in enumerate(footsteps):
        pos = footstep.position[:2]
        side = 'Right' if footstep.side else 'Left'
        color = 'red' if footstep.side else 'blue'
        marker = 's' if footstep.side else 'o'
        ax1.plot(pos[0], pos[1], marker, color=color, markersize=10, label=f'{side} Foot')
    
    # Plot DCM trajectory
    for i, step_dcm in enumerate(dcm_traj):
        dcm_points = np.array(step_dcm)
        ax1.plot(dcm_points[:, 0], dcm_points[:, 1], 'g-', linewidth=2, alpha=0.8, label=f'DCM Step {i+1}' if i == 0 else "")
    
    # Plot CoM trajectory
    for i, step_com in enumerate(com_traj):
        com_points = np.array(step_com)
        ax1.plot(com_points[:, 0], com_points[:, 1], 'm-', linewidth=2, alpha=0.8, label=f'CoM Step {i+1}' if i == 0 else "")
    
    # Plot ZMP trajectory
    for i, step_zmp in enumerate(zmp_traj):
        zmp_points = np.array(step_zmp)
        ax1.plot(zmp_points[:, 0], zmp_points[:, 1], 'c-', linewidth=2, alpha=0.8, label=f'ZMP Step {i+1}' if i == 0 else "")
    
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: DCM vs CoM Comparison (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('DCM vs CoM Relationship', fontsize=12, fontweight='bold')
    
    for i, (step_dcm, step_com) in enumerate(zip(dcm_traj, com_traj)):
        dcm_points = np.array(step_dcm)
        com_points = np.array(step_com)
        
        ax2.plot(dcm_points[:, 0], dcm_points[:, 1], 'g-', linewidth=2, alpha=0.7, label=f'DCM Step {i+1}')
        ax2.plot(com_points[:, 0], com_points[:, 1], 'm--', linewidth=2, alpha=0.7, label=f'CoM Step {i+1}')
    
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: DCM vs ZMP Comparison (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('DCM vs ZMP Relationship', fontsize=12, fontweight='bold')
    
    for i, (step_dcm, step_zmp) in enumerate(zip(dcm_traj, zmp_traj)):
        dcm_points = np.array(step_dcm)
        zmp_points = np.array(step_zmp)
        
        ax3.plot(dcm_points[:, 0], dcm_points[:, 1], 'g-', linewidth=2, alpha=0.7, label=f'DCM Step {i+1}')
        ax3.plot(zmp_points[:, 0], zmp_points[:, 1], 'c--', linewidth=2, alpha=0.7, label=f'ZMP Step {i+1}')
    
    ax3.set_xlabel('X Position (m)')
    ax3.set_ylabel('Y Position (m)')
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: X Position vs Time (bottom left)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_title('X Position vs Time', fontsize=12, fontweight='bold')
    
    # Use the shortest trajectory length to ensure compatibility
    min_length = min(len(dcm_traj[0]), len(com_traj[0]), len(zmp_traj[0]))
    time_points = np.linspace(0, step_duration, min_length)
    
    for i, (step_dcm, step_com, step_zmp) in enumerate(zip(dcm_traj, com_traj, zmp_traj)):
        # Truncate all trajectories to the same length
        dcm_x = [point[0] for point in step_dcm[:min_length]]
        com_x = [point[0] for point in step_com[:min_length]]
        zmp_x = [point[0] for point in step_zmp[:min_length]]
        
        ax4.plot(time_points, dcm_x, 'g-', linewidth=2, alpha=0.7, label=f'DCM Step {i+1}')
        ax4.plot(time_points, com_x, 'm--', linewidth=2, alpha=0.7, label=f'CoM Step {i+1}')
        ax4.plot(time_points, zmp_x, 'c:', linewidth=2, alpha=0.7, label=f'ZMP Step {i+1}')
    
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('X Position (m)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Plot 5: Y Position vs Time (bottom middle)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_title('Y Position vs Time', fontsize=12, fontweight='bold')
    
    for i, (step_dcm, step_com, step_zmp) in enumerate(zip(dcm_traj, com_traj, zmp_traj)):
        # Truncate all trajectories to the same length
        dcm_y = [point[1] for point in step_dcm[:min_length]]
        com_y = [point[1] for point in step_com[:min_length]]
        zmp_y = [point[1] for point in step_zmp[:min_length]]
        
        ax5.plot(time_points, dcm_y, 'g-', linewidth=2, alpha=0.7, label=f'DCM Step {i+1}')
        ax5.plot(time_points, com_y, 'm--', linewidth=2, alpha=0.7, label=f'CoM Step {i+1}')
        ax5.plot(time_points, zmp_y, 'c:', linewidth=2, alpha=0.7, label=f'ZMP Step {i+1}')
    
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Y Position (m)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Plot 6: Stability Analysis (bottom right)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_title('Stability Analysis: DCM-CoM Distance', fontsize=12, fontweight='bold')
    
    for i, (step_dcm, step_com) in enumerate(zip(dcm_traj, com_traj)):
        distances = []
        # Use the shorter length to avoid index errors
        step_length = min(len(step_dcm), len(step_com))
        for j in range(step_length):
            dcm_point = step_dcm[j]
            com_point = step_com[j]
            distance = np.linalg.norm(np.array(dcm_point) - np.array(com_point))
            distances.append(distance)
        
        # Create time points for this step
        step_time_points = np.linspace(0, step_duration, len(distances))
        ax6.plot(step_time_points, distances, linewidth=2, alpha=0.7, label=f'Step {i+1}')
    
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('DCM-CoM Distance (m)')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.suptitle('Complete Bipedal Walking Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("\n" + "="*50)
    print("DETAILED TRAJECTORY ANALYSIS")
    print("="*50)
    
    print(f"\nTRAJECTORY STATISTICS:")
    print(f"   • Footsteps: {len(footsteps)}")
    print(f"   • DCM trajectory steps: {len(dcm_traj)}")
    print(f"   • CoM trajectory steps: {len(com_traj)}")
    print(f"   • ZMP trajectory steps: {len(zmp_traj)}")
    print(f"   • Time step: {dt:.3f}s")
    print(f"   • Step duration: {step_duration:.1f}s")
    
    # Print trajectory lengths for debugging
    if dcm_traj and com_traj and zmp_traj:
        print(f"   • DCM points per step: {len(dcm_traj[0])}")
        print(f"   • CoM points per step: {len(com_traj[0])}")
        print(f"   • ZMP points per step: {len(zmp_traj[0])}")
        print(f"   • Used points per step: {min_length}")
    
    print(f"\nKEY PARAMETERS:")
    print(f"   • Step width: {step_width:.3f}m")
    print(f"   • Step length: {step_length:.3f}m")
    print(f"   • Natural frequency (ω): {np.sqrt(9.81/0.4):.2f} rad/s")
    print(f"   • CoM height: 0.4m")
    
    print(f"\nTRAJECTORY POINTS:")
    if dcm_traj and dcm_traj[0]:
        print(f"   • DCM Step 1, Start: {dcm_traj[0][0]}")
        print(f"   • DCM Step 1, End: {dcm_traj[0][-1]}")
    if com_traj and com_traj[0]:
        print(f"   • CoM Step 1, Start: {com_traj[0][0]}")
        print(f"   • CoM Step 1, End: {com_traj[0][-1]}")
    
    # Calculate stability metrics
    print(f"\nSTABILITY ANALYSIS:")
    if dcm_traj and com_traj:
        total_distance = 0
        count = 0
        for step_dcm, step_com in zip(dcm_traj, com_traj):
            step_length = min(len(step_dcm), len(step_com))
            for j in range(step_length):
                dcm_point = step_dcm[j]
                com_point = step_com[j]
                distance = np.linalg.norm(np.array(dcm_point) - np.array(com_point))
                total_distance += distance
                count += 1
        
        if count > 0:
            avg_distance = total_distance / count
            print(f"   • Average DCM-CoM distance: {avg_distance:.4f}m")
            
            # Calculate max distance
            max_distance = 0
            for step_dcm, step_com in zip(dcm_traj, com_traj):
                step_length = min(len(step_dcm), len(step_com))
                for j in range(step_length):
                    dcm_point = step_dcm[j]
                    com_point = step_com[j]
                    distance = np.linalg.norm(np.array(dcm_point) - np.array(com_point))
                    max_distance = max(max_distance, distance)
            print(f"   • Max DCM-CoM distance: {max_distance:.4f}m")
    
    print("\nAll trajectories generated successfully!")
    return True

if __name__ == "__main__":
    test_complete_library()