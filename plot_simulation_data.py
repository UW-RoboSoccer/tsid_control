
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_simulation_data(com_log_path, debug_log_path):
    # Load true CoM log
    true_com_log = np.load(com_log_path)

    # Load debug log
    debug_df = pd.read_csv(debug_log_path)

    # Extract relevant data from debug_df
    sim_time = debug_df['sim_time'].values
    com_ref_x = debug_df['com_ref_x'].values
    com_ref_y = debug_df['com_ref_y'].values
    com_ref_z = debug_df['com_ref_z'].values
    left_foot_x = debug_df['left_foot_x'].values
    left_foot_y = debug_df['left_foot_y'].values
    right_foot_x = debug_df['right_foot_x'].values
    right_foot_y = debug_df['right_foot_y'].values

    # Plot 1: CoM Trajectories (X-Y plane)
    plt.figure(figsize=(10, 8))
    plt.plot(true_com_log[:, 0], true_com_log[:, 1], label='True CoM (Sim)', alpha=0.7)
    plt.plot(com_ref_x, com_ref_y, label='Reference CoM', linestyle='--', alpha=0.7)
    plt.scatter(left_foot_x[::100], left_foot_y[::100], color='green', marker='o', s=10, label='Left Foot (sampled)')
    plt.scatter(right_foot_x[::100], right_foot_y[::100], color='red', marker='x', s=10, label='Right Foot (sampled)')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('CoM Trajectories and Foot Positions (X-Y Plane)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig('com_trajectories_xy.png')
    plt.close()

    # Plot 2: CoM Height (Z-axis) over time
    plt.figure(figsize=(10, 6))
    plt.plot(sim_time, true_com_log[:, 2], label='True CoM Height (Sim)', alpha=0.7)
    plt.plot(sim_time, com_ref_z, label='Reference CoM Height', linestyle='--', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Z Position (m)')
    plt.title('CoM Height Over Time')
    plt.grid(True)
    plt.legend()
    plt.savefig('com_height_z.png')
    plt.close()

    print(f"Plots saved to {os.path.abspath('com_trajectories_xy.png')} and {os.path.abspath('com_height_z.png')}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    com_log_path = os.path.join(script_dir, 'true_com_log.npy')
    debug_log_path = os.path.join(script_dir, 'walking_debug_log.csv')

    if os.path.exists(com_log_path) and os.path.exists(debug_log_path):
        plot_simulation_data(com_log_path, debug_log_path)
    else:
        print("Error: Data files not found. Please ensure 'true_com_log.npy' and 'walking_debug_log.csv' exist in the current directory.")
        print(f"Looking for: {com_log_path} and {debug_log_path}")
