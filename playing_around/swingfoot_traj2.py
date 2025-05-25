import numpy as np
import matplotlib.pyplot as plt



# Parameters
step_length = 0.1
step_time = 2
step_height = 0.2



# Time array from 0 to 2 with step of 0.01
t_values = np.arange(0, 2.01, 0.01)



# Initialize arrays
xs = np.zeros_like(t_values)
zs = np.zeros_like(t_values)
zs_dif = np.zeros_like(t_values)
buff = 0



# Compute xs, zs, zs_dif
for ind, t in enumerate(t_values):
    xs[ind] = (-step_length * np.cos(np.pi * t / step_time)) + step_length
    zs[ind] = 0.5 * step_height * (1 - np.cos(2 * np.pi * t / step_time))
    zs_dif[ind] = zs[ind] - buff
    buff = zs[ind]



# Plotting z and x
plt.figure()
plt.plot(zs, label='z')
plt.plot(xs, label='x')
plt.legend()
plt.title('Z and X over time')
plt.xlabel('Time step index')
plt.ylabel('Position')



# Plotting x vs z
plt.figure()
plt.plot(xs, zs)
plt.xlabel('x')
plt.ylabel('z')
plt.title('Foot Trajectory')



# Plotting zs_dif
plt.figure()
plt.plot(zs_dif)
plt.title('Change in Z (zs_dif)')
plt.xlabel('Time step index')
plt.ylabel('Δz')



plt.tight_layout()
plt.show()