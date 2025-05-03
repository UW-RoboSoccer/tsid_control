import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

class LinearInvertedPendulum:
    def __init__(self, height=1.0, g=9.81):
        self.h = height  # pendulum height
        self.g = g       # gravity
        self.omega = np.sqrt(self.g / self.h)  # natural frequency
        
        # State: [x, x_dot]
        self.state = np.zeros(2)
        self.zmp = 0.0
        
    def set_state(self, position, velocity):
        self.state[0] = position
        self.state[1] = velocity
        
    def compute_capture_point(self):
        # ξ = x + ẋ/ω
        return self.state[0] + self.state[1] / self.omega
    
    def set_zmp(self, zmp):
        self.zmp = zmp
        
    def step(self, dt):
        # ẍ = ω² (x - p)
        x_ddot = self.omega**2 * (self.state[0] - self.zmp)
        
        # Update state using semi-implicit Euler integration
        self.state[1] += x_ddot * dt  # update velocity
        self.state[0] += self.state[1] * dt  # update position
        
        return self.state.copy()

# Simulation parameters
sim_time = 5.0  # seconds
dt = 0.01       # timestep
steps = int(sim_time / dt)

# Create pendulum and set initial state
pendulum = LinearInvertedPendulum(height=0.8)
initial_pos = 0.5    # initial position (meters)
initial_vel = 1.0    # initial velocity (m/s)
pendulum.set_state(initial_pos, initial_vel)

# Storage for simulation data
time_data = np.linspace(0, sim_time, steps)
com_data = np.zeros((steps, 2))   # [pos, vel]
zmp_data = np.zeros(steps)
cp_data = np.zeros(steps)

# Perform simulation
capture_point = pendulum.compute_capture_point()
print(f"Initial capture point: {capture_point:.3f} m")

for i in range(steps):
    # Calculate capture point
    cp = pendulum.compute_capture_point()
    
    # Move ZMP to capture point (in a real robot, this would be done by shifting the feet)
    pendulum.set_zmp(cp)
    
    # Store data
    com_data[i] = pendulum.state
    zmp_data[i] = pendulum.zmp
    cp_data[i] = cp
    
    # Simulate one timestep
    pendulum.step(dt)

print (f"Simulation completed.")

# Create visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the position data
ax.plot(time_data, com_data[:, 0], 'b-', label='CoM Position')
ax.plot(time_data, zmp_data, 'r--', label='ZMP Position')
ax.plot(time_data, cp_data, 'g-.', label='Capture Point')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (m)')
ax.legend()
ax.set_title('Linear Inverted Pendulum with Capture Point Control')
ax.grid(True)

# Create animation of the pendulum
fig_anim, ax_anim = plt.subplots(figsize=(10, 6))
ax_anim.set_xlim(-1, 1.5)
ax_anim.set_ylim(0, 1.5)
ax_anim.set_xlabel('Position (m)')
ax_anim.set_ylabel('Height (m)')
ax_anim.set_title('Linear Inverted Pendulum Animation')
ax_anim.grid(True)

# Create pendulum and support polygon elements
pendulum_line, = ax_anim.plot([], [], 'k-', lw=2)
com_point, = ax_anim.plot([], [], 'bo', ms=10, label='CoM')
zmp_point, = ax_anim.plot([], [], 'ro', ms=8, label='ZMP')
cp_point, = ax_anim.plot([], [], 'go', ms=8, label='Capture Point')
ground_line, = ax_anim.plot([-1, 1.5], [0, 0], 'k-', lw=1)

time_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes)
ax_anim.legend(loc='upper right')

def init():
    pendulum_line.set_data([], [])
    com_point.set_data([], [])
    zmp_point.set_data([], [])
    cp_point.set_data([], [])
    time_text.set_text('')
    return pendulum_line, com_point, zmp_point, cp_point, time_text

def animate(i):
    x = com_data[i, 0]
    zmp = zmp_data[i]
    cp = cp_data[i]
    
    # Update pendulum visualization
    pendulum_line.set_data([zmp, x], [0, pendulum.h])
    com_point.set_data([x], [pendulum.h])
    zmp_point.set_data([zmp], [0])
    cp_point.set_data([cp], [0]) 
    time_text.set_text(f'Time: {time_data[i]:.2f}s')
    
    return pendulum_line, com_point, zmp_point, cp_point, time_text

# Create animation
ani = FuncAnimation(fig_anim, animate, frames=range(0, steps, 2),
                   init_func=init, blit=True, interval=20)

plt.tight_layout()
plt.show()

# Optionally save the animation
# ani.save('capture_point_simulation.mp4', writer='ffmpeg', fps=30)