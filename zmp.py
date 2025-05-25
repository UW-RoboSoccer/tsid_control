import numpy as np
import op3_conf as conf
# import op3_conf as conf
# import /../op3_conf as conf   
# from biped import Biped
import matplotlib.pyplot as plt

mass = 5
g = conf.g
w = conf.d_bw_feet
step_height = 0.2
step_length = 0.1

N_steps = 6
M = 50 #samples per step

foot_positions = np.zeros((N_steps, 2))

for i in range (0, N_steps):
    sign = (-1)**i
    foot_positions[i] = [i * step_length, sign * w]

# Plot 2D placements for reference
plt.figure()
plt.scatter(foot_positions[:,0], foot_positions[:,1], c='k', marker='x')
plt.title("2D Foot Placements")
plt.xlabel("x [m]")
plt.ylabel("y [m]")

# Prepare 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Swing-Foot Trajectories")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")


# Generate and plot each swing trajectory
for i in range(N_steps - 1):
    # Parameter t
    t = np.linspace(0, 1, M)

    # Cubic spline interpolation for x, y
    xs = np.linspace(foot_positions[i,0], foot_positions[i+1,0], M)
    ys = np.linspace(foot_positions[i,1], foot_positions[i+1,1], M)

    # Parabolic z-profile
    zs = 4 * step_height * t * (1 - t)

    # Plot
    ax.plot(xs, ys, zs)

plt.show()




# t_values = np.arange(0, 2.01, 0.01)

# T = 10 #seconds
# dt = 0.010 #milliseconds

# num_points = T/dt

# #initial positions
# pos_x = 0.0
# pos_y = 0.0
# pos_z = 0.0
# theta = 0.0



# def find_zmp(accel_array, pos_array):
#     tau_x = mass * accel_array[0] * pos_array[2]
#     tau_y = mass * accel_array[1] * pos_array[2]
#     zmp_x = -tau_x / (mass * g)
#     zmp_y = -tau_y / (mass * g)

#     return (zmp_x, zmp_y)

# zx, zy = find_zmp([1.0, 2.0], [0, 0, conf.z_com])
