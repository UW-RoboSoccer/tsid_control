import numpy as np
from scipy.interpolate import CubicSpline

class Controller:
    """Bipedal walking controller using Divergent Component of Motion (DCM) approach.
    
    This controller generates footstep trajectories, Zero Moment Point (ZMP) references,
    and Divergent Component of Motion (DCM) trajectories for stable bipedal walking.
    
    Attributes:
        biped: Biped robot model
        conf: Configuration parameters
        dt: Time step
        sim_time: Total simulation time
        steps: Number of simulation steps
        current_step: Current footstep index
        w_n: Natural frequency of the linearized inverted pendulum
        x: Center of Mass (CoM) position [x, y, z]
        dx: CoM velocity [dx, dy, dz]
        ddx: CoM acceleration [ddx, ddy, ddz]
        e: DCM position [x, y, z]
        de: DCM velocity [dx, dy, dz]
        zmp: Zero Moment Point position [x, y]
        footstep_traj: List of footstep trajectories
        zmp_traj: List of ZMP trajectory points
        dcm_traj: List of DCM trajectory points
        com_traj: List of CoM trajectory points
    """
    def __init__(self, biped, conf):
        """Initialize the controller with the biped model and configuration parameters.
        
        Args:
            biped: Biped robot model
            conf: Configuration parameters
        """
        self.biped = biped
        self.conf = conf
        self.dt = conf.dt
        self.time_step = 0
        self.current_step = 0
        self.depth = 3  # Number of future steps to consider for trajectory generation
        
        # Natural frequency of the linearized inverted pendulum
        self.w_n = np.sqrt(conf.z0 / conf.g)
        
        # Initialize state variables
        self.x = np.zeros(3)     # [x, y, z] CoM position
        self.dx = np.zeros(3)    # [dx, dy, dz] CoM velocity
        self.ddx = np.zeros(3)   # [ddx, ddy, ddz] CoM acceleration
        self.e = np.zeros(3)     # [x, y, z] DCM position
        self.de = np.zeros(3)    # [dx, dy, dz] DCM velocity
        self.zmp = np.zeros(2)   # [x, y] ZMP position
        
        # Initialize trajectories
        self.footsteps = []  # Footstep positions
        self.footstep_traj = []  # Footstep trajectory
        self.zmp_traj = []       # ZMP trajectory
        self.dcm_traj = []       # DCM trajectory
        self.com_traj = []       # CoM trajectory
        
    def gen_footsteps(self, traj, orientation):
        """Generate footstep positions and trajectories from a reference path.
        
        Generates alternating left/right footsteps along a reference path with
        specified step length and width. Each footstep includes position,
        orientation, and foot side (left/right).
        
        Args:
            traj: Reference path as a numpy array of shape (n, 2) for x,y coordinates
        """
        self.current_step = 0
        footsteps = []
        dist = 0
        right_foot =  True # Start with right foot

        tangent = orientation
        normal = np.array([-tangent[1], tangent[0]])
        # Initial footstep at origin
        footsteps.append(np.array([self.conf.step_width * normal[0] * -1, self.conf.step_width * normal[1] * -1, 0, False]))  # Initial footstep at origin

        # Generate discrete footstep positions
        for i in range(len(traj) - 1):
            # Calculate path segment direction
            dx = traj[i + 1, 0] - traj[i, 0]
            dy = traj[i + 1, 1] - traj[i, 1]

            # Accumulate distance
            dist += np.sqrt(dx**2 + dy**2)
            
            # Place a footstep when we've traveled the step length
            if dist >= self.conf.step_length:
                # Calculate tangent and normal vectors to the path
                tangent = np.array([dx, dy])
                tangent /= np.linalg.norm(tangent)
                normal = np.array([-tangent[1], tangent[0]])  # Perpendicular to tangent
                
                # Create footstep: [x, y, yaw, is_right_foot]
                footstep = np.zeros(4)
                footstep[0] = traj[i, 0] + self.conf.step_width * normal[0] * (1 if right_foot else -1)
                footstep[1] = traj[i, 1] + self.conf.step_width * normal[1] * (1 if right_foot else -1)
                footstep[2] = np.arctan2(tangent[1], tangent[0])  # Yaw angle
                footstep[3] = right_foot
                footsteps.append(footstep)
                
                # Reset distance and alternate feet
                dist = 0
                right_foot = not right_foot
    
        # Generate smooth trajectories between footsteps
        footstep_traj = []
        num_points = int(self.conf.step_time / self.conf.dt)
        t = np.linspace(0, 1, num_points)
        
        for i in range(len(footsteps) - 2):
            p0 = footsteps[i][:2]    # Start position (x,y)
            p1 = footsteps[i + 2][:2]  # End position (x,y)
            
            # Create cubic spline for x-y trajectory
            t_spline = np.array([0, 1])
            x_control = np.array([p0[0], p1[0]])
            y_control = np.array([p0[1], p1[1]])
            
            x_traj = CubicSpline(t_spline, x_control)(t)
            y_traj = CubicSpline(t_spline, y_control)(t)
            
            # Create parabolic height profile for smooth foot lifting
            z_traj = 4 * self.conf.step_height * t * (1 - t)

            # Create homogeneous transformation matrices for each point
            HTM = []
            yaw = footsteps[i + 1][2]  # Use the yaw of the intermediate footstep
            for j in range(num_points):
                # Create rotation matrix from yaw angle
                T = np.eye(4)
                R = np.array([
                    [np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]
                ])
                T[:3, :3] = R
                T[:3, 3] = [x_traj[j], y_traj[j], z_traj[j]]
                HTM.append(T)

            footstep_traj.append((footsteps[i][3], HTM))

        self.footstep_traj = footstep_traj
        self.footsteps = footsteps

        return footsteps

    def gen_dcm_traj(self, depth=3):
        """Generate DCM trajectory by backward recursion from final footstep.
        
        Creates a Divergent Component of Motion (DCM) trajectory by calculating
        backwards from a terminal DCM position at the final footstep, ensuring
        DCM convergence to each Virtual Repellent Point (VRP) position.
        
        Args:
            depth: Number of future steps to consider
        """
        # Initialize DCM endpoints array
        dcm_endpoints = []
        
        # Start from the VRP (Virtual Repellent Point) at the last footstep
        vrp_pos = self.footsteps[depth - 1][:2]
        dcm_end = vrp_pos  # Terminal DCM equals the terminal VRP

        # Add final DCM endpoint
        dcm_endpoints.append(dcm_end)

        # Calculate DCM endpoints by backward recursion
        for i in range(depth-2, 0, -1):
            # Get VRP position (foot position)
            vrp_pos = self.footsteps[i + self.current_step][:2]
            
            # Calculate initial DCM for this step to reach dcm_end at the end of the step
            dcm_i = self.back_calc_dcm(vrp_pos, dcm_end)
            
            # Update for next iteration
            dcm_end = dcm_i
            dcm_endpoints.insert(0, dcm_i)  # Insert at beginning to maintain order

        # Generate intermediate DCM points with exponential curves
        for i in range(len(dcm_endpoints)):
            dcm_end = dcm_endpoints[i]  # Final DCM for this step
            vrp = self.footsteps[i + self.current_step][:2]

            dcm_inter_step = []  # List to hold DCM points for this step

            # Generate DCM points for this step
            for j in range(int(self.conf.step_time / self.dt)):
                t = j * self.dt
                
                # DCM evolution follows: ξ(t) = p + e^(t/T_c)*(ξ_0 - p)
                # where p is the VRP position and ξ_0 is the initial DCM
                dcm_t = vrp + (dcm_end - vrp) * np.exp((t - self.conf.step_time) / self.w_n)
                dcm_inter_step.append(dcm_t)

            self.dcm_traj.append(dcm_inter_step)  # Append the DCM points for this step

        # Add the final DCM point
        self.dcm_traj[-1].append(dcm_endpoints[-1])

        return dcm_endpoints

    def back_calc_dcm(self, vrp_pos, dcm_end):
        """Calculate the initial DCM position to reach a desired DCM endpoint.
        
        Uses the exponential DCM dynamics to find the initial DCM position that will
        naturally evolve to the desired endpoint after a step duration.
        
        Args:
            vrp_pos: Virtual Repellent Point position [x, y]
            dcm_end: Desired DCM endpoint [x, y]
            
        Returns:
            Initial DCM position [x, y]
        """
        # DCM evolution is: ξ(t) = p + e^(t/T_c)*(ξ_0 - p)
        # Solving for ξ_0: ξ_0 = p + e^(-t/T_c)*(ξ(t) - p)
        dcm_ini = vrp_pos + (dcm_end - vrp_pos) * np.exp(-self.conf.step_time / self.w_n)
        return dcm_ini

    def dcm_controller(self, dcm_ref, vrp_i):
        """DCM feedback controller to calculate ZMP command and external force.
        
        Implements a feedback controller for DCM tracking, calculating the required
        ZMP reference to drive the DCM toward the reference.
        
        Args:
            dcm_ref: Reference DCM position [x, y, z]
            vrp_i: Initial vrp position [x, y, z]
            
        Returns:
            com: Center of Mass (CoM) state [x, dx, ddx]
            F_ext: External force required [Fx, Fy, Fz]
        """
        # DCM feedback control law:
        # p = ξ + T_c*ξ̇ - T_c*k_ξ*(ξ - ξ_ref) - T_c*ξ̇_ref
        vrp_control = vrp_i + (1 + self.conf.k_dcm * self.w_n) * (self.e - dcm_ref)
        
        # Convert VRP to ZMP by subtracting pendulum height in z direction
        zmp_control = vrp_control - np.array([0, 0, self.conf.z0])
        ddx = (1 / self.w_n**2) * (self.x - zmp_control)
        dx = self.dx + ddx * self.dt
        x = self.x + dx * self.dt

        com = np.array([x, dx, ddx])

        # Calculate required external force using the ZMP equation
        F_ext = (self.conf.m / (self.w_n**2)) * (self.x - zmp_control)

        return com, F_ext
    
    def gen_com_traj(self, x0, dx0):
        """Generate CoM trajectory based on DCM trajectory.
        
        This method generates the CoM trajectory by assuming a LIPM model
        and a constant height for the CoM. The CoM trajectory is generated
        using the ZMP reference.
        """
        
        ddx = np.zeros(2)
        dx = dx0
        x = x0
        for zmp_path in self.zmp_traj:
            # Calculate the CoM trajectory using the ZMP reference
            com_traj = []
            for t in zmp_path:
                ddx = (1 / self.w_n**2) * (x - t)
                dx = dx + ddx * self.dt
                x = x + dx * self.dt
                com_traj.append(x)
                
            self.com_traj.append(com_traj)

    def gen_zmp_traj(self):
        """Generate ZMP trajectory based on footstep trajectory.
        
        This method generates the ZMP trajectory by interpolating between
        the footstep positions using a cubic spline. The ZMP trajectory is generated using the
        footstep trajectory.
        """

        # Initialize the ZMP trajectory
        self.zmp_traj = []
        
        # If we don't have enough footsteps, return
        if len(self.footsteps) < 2:
            return
        
        # For each adjacent pair of footsteps
        for i in range(len(self.footsteps) - 1):
            # Get the positions of the current and next footstep
            p0 = self.footsteps[i][:2]    # Start position (x,y)
            p1 = self.footsteps[i + 1][:2]  # End position (x,y)
            
            # Create cubic spline for x-y trajectory
            num_points = int(self.conf.step_time / self.conf.dt)
            t = np.linspace(0, 1, num_points)
            t_spline = np.array([0, 1])
            
            x_control = np.array([p0[0], p1[0]])
            y_control = np.array([p0[1], p1[1]])
            
            x_traj = CubicSpline(t_spline, x_control)(t)
            y_traj = CubicSpline(t_spline, y_control)(t)
            
            # Add the interpolated points to the ZMP trajectory
            step_zmp = []
            for j in range(num_points):
                step_zmp.append(np.array([x_traj[j], y_traj[j]]))
            
            self.zmp_traj.append(step_zmp)
    
    def update(self):
        """Update the controller state and compute the next control commands.
        
        This method is called at each simulation step to update the controller
        state and compute the next control commands for the biped robot.
        """
        # Update trajectories
        if self.current_step % self.depth == 0:
            self.gen_dcm_traj()
        
        # Get current DCM and ZMP references
        dcm_ref = self.dcm_traj[self.time_step]
        vrp_ref = self.footstep_traj[self.current_step][1][0][:3, 3] + self.conf.z0 * np.array([0, 0, 1])
        
        # Calculate DCM feedback control
        com_control, F_ext = self.dcm_controller(dcm_ref, vrp_ref)

        # Set CoM task references
        self.biped.sample_com.pos(com_control[0])
        self.biped.sample_com.vel(com_control[1])
        self.biped.sample_com.acc(com_control[2])
        self.biped.trajCom.setReference(self.biped.sample_com)
        
        # Set foot position
        footstep = self.footstep_traj[self.current_step]
        if footstep[0]: # Right foot
            self.biped.sample_RF.value(footstep[1][self.time_step][:3, 3])
            self.biped.trajRF.setReference(self.biped.sample_RF)
        else: # Left foot
            self.biped.sample_LF.value(footstep[1][self.time_step][:3, 3])
            self.biped.trajLF.setReference(self.biped.sample_LF)

        # Update current step index
        self.time_step += 1
        if self.time_step >= self.conf.step_time / self.dt:
            self.time_step = 0
            self.current_step += 1
            if self.current_step >= len(self.footstep_traj):
                raise StopIteration("End of footstep trajectory")

        return

    def reset(self):
        """Reset the controller state for a new trajectory."""
        self.time_step = 0
        self.current_step = 0
        self.footstep_traj = []
        self.dcm_traj = []
        self.com_traj = []

        # Reset state variables
        self.x = np.zeros(3)
        self.dx = np.zeros(3)
        self.ddx = np.zeros(3)
        self.e = np.zeros(3)
        self.de = np.zeros(3)
        self.zmp = np.zeros(2)

    def plot_path(self, ax):
        """Plot the footstep trajectory and DCM trajectory on a 2D plot.
        
        Args:
            ax: Matplotlib axis object to plot on
        """
        # Plot footstep trajectory
        for footstep in self.footstep_traj:
            foot = footstep[0]
            traj = footstep[1]
            print(foot)
            for t in traj:
                ax.plot(t[0, 3], t[1, 3], 'ro' if foot else 'bo')
        
        # Plot DCM trajectory
        for dcm in self.dcm_traj:
            for t in dcm:
                ax.plot(t[0], t[1], 'go')

        # Plot CoM trajectory
        for com in self.com_traj:
            for t in com:
                ax.plot(t[0], t[1], 'yo')

        # Plot ZMP trajectory
        for zmp in self.zmp_traj:
            for t in zmp:
                ax.plot(t[0], t[1], 'co')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Footstep and DCM Trajectory')
        ax.axis('equal')
        ax.grid()

import matplotlib.pyplot as plt
import numpy as np

import op3_conf as conf
from biped import Biped

# Initialize the Biped robot model
biped = Biped(conf)
# Initialize the controller
controller = Controller(biped, conf)
# Generate a reference path based on linear and angular velocities
v = 1.0
w = 0.0
T = conf.N_SIMULATION * conf.dt
dt = conf.dt

# Initial position
x = 0.0
y = 0.0

# Generate reference trajectory
t = np.linspace(0, T, int(T / dt))
traj = np.zeros((len(t), 2))
for i in range(len(t)):
    x = x + v * dt * np.cos(w)
    y = y + v * dt * np.sin(w)
    traj[i, :] = [x, y]

initial_orientation = np.array([np.cos(w), np.sin(w)])  # Initial orientation vector

# Generate footstep trajectory
gen_foosteps = controller.gen_footsteps(traj, initial_orientation)
print("Footsteps generated:", gen_foosteps)

# Generate DCM trajectory
gen_dcm_endpoints = controller.gen_dcm_traj(5)
print("DCM endpoints generated:", gen_dcm_endpoints)

# Generate ZMP trajectory
controller.gen_zmp_traj()
print("ZMP trajectory generated:", controller.zmp_traj)

# Generate CoM trajectory
controller.gen_com_traj(np.array([0, -0.06]), np.array([0.2, 0]))

# Initialize plot
fig, ax = plt.subplots()

# Plot the footstep and DCM trajectories
controller.plot_path(ax)

# Plot the footsteps
for footstep in gen_foosteps:
    foot = footstep[3]
    ax.plot(footstep[0], footstep[1], 'mo')

# Plot the DCM endpoints
for dcm in gen_dcm_endpoints:
    ax.plot(dcm[0], dcm[1], 'co')

# set axis limits
ax.set_xlim(0, .5)
ax.set_ylim(-.5, .5)

# Show the plot
plt.show()