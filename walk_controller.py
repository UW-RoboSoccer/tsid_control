import numpy as np
from scipy.interpolate import CubicSpline
import pinocchio as pin

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
        logged_com_traj: List of logged CoM positions
        swing_foot: Current swing foot ('left' or 'right' or None)
        last_step: Index of the last processed step
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
        self.logged_com_traj = []
        
        # Initialize rate limiting variables
        self.prev_dcm_ref = None
        self.prev_com_ref = None
        
        self.swing_foot = None  # 'left' or 'right' or None
        self.last_step = -1
        
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
        
        # Initial footstep positions - more conservative placement
        left_init = np.array([-self.conf.step_width/2, 0, 0, False])  # Left foot
        right_init = np.array([self.conf.step_width/2, 0, 0, True])   # Right foot
        footsteps.append(left_init)
        footsteps.append(right_init)

        # Generate discrete footstep positions with more conservative approach
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
                if np.linalg.norm(tangent) > 1e-6:
                    tangent /= np.linalg.norm(tangent)
                else:
                    tangent = np.array([1.0, 0.0])  # Default forward direction
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
    
        # Generate smooth trajectories between footsteps with better constraints
        footstep_traj = []
        num_points = int(self.conf.step_time / self.conf.dt)
        t = np.linspace(0, 1, num_points)
        
        for i in range(len(footsteps) - 2):
            p0 = footsteps[i][:2]    # Start position (x,y)
            p1 = footsteps[i + 2][:2]  # End position (x,y)
            
            # Limit maximum step distance for stability
            step_dist = np.linalg.norm(p1 - p0)
            if step_dist > 0.05:  # More conservative step distance limit
                direction = (p1 - p0) / step_dist
                p1 = p0 + 0.05 * direction
            
            # Create cubic spline for x-y trajectory
            t_spline = np.array([0, 1])
            x_control = np.array([p0[0], p1[0]])
            y_control = np.array([p0[1], p1[1]])
            
            x_traj = CubicSpline(t_spline, x_control)(t)
            y_traj = CubicSpline(t_spline, y_control)(t)
            
            # Create smoother height profile for better foot clearance
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

        # Print first 10 footsteps for debugging
        print("First 10 footsteps:")
        for i in range(min(10, len(footsteps))):
            print(footsteps[i])

        return footsteps

    def gen_dcm_traj(self):
        """Generate the DCM trajectory for the entire footstep plan."""
        self.dcm_traj = []
        dcm_endpoints = []
        N = len(self.footsteps)
        if N < 2:
            return []
        # Set the final DCM endpoint at the last footstep
        dcm_end = self.footsteps[-1][:2]
        dcm_endpoints = [dcm_end]
        # Backward recursion for all steps
        for i in range(N-2, -1, -1):
            vrp_pos = self.footsteps[i][:2]
            dcm_i = self.back_calc_dcm(vrp_pos, dcm_end)
            dcm_end = dcm_i
            dcm_endpoints.insert(0, dcm_i)
        # Generate DCM segments for all steps
        for i in range(len(dcm_endpoints) - 1):
            dcm_start = dcm_endpoints[i]
            dcm_end = dcm_endpoints[i + 1]
            vrp = self.footsteps[i][:2]
            dcm_inter_step = []
            num_points = int(self.conf.step_time / self.dt)
            for j in range(num_points):
                t = j * self.dt
                # Use linear interpolation between DCM start and end points
                # This ensures the DCM actually moves forward
                alpha = t / self.conf.step_time
                dcm_t = dcm_start + alpha * (dcm_end - dcm_start)
                dcm_inter_step.append(dcm_t)
            self.dcm_traj.append(dcm_inter_step)
        # Print first 10 DCM trajectory points for debugging
        print("First 10 DCM trajectory points (first step):")
        for i in range(min(10, len(self.dcm_traj[0]))):
            print(self.dcm_traj[0][i])
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
        """DCM feedback controller to calculate CoM reference and external force."""
        # Get current robot state from biped
        data = self.biped.formulation.data()
        current_com = self.biped.robot.com(data)
        current_com_vel = self.biped.robot.com_vel(data)
        
        # Update controller state variables
        self.x = current_com
        self.dx = current_com_vel
        
        # Compute current DCM (ξ = x + ẋ/ω)
        self.e = self.x + self.dx / self.w_n
        
        # Very conservative DCM feedback control law:
        com_ref = dcm_ref.copy()
        com_ref = np.array([dcm_ref[0], dcm_ref[1], self.conf.z0])  # Keep height constant
        
        # Very conservative damping - stay close to current position
        damping_factor_x = 0.1  # Very small movement in x
        damping_factor_y = 0.05  # Very small movement in y
        com_ref[0] = damping_factor_x * com_ref[0] + (1 - damping_factor_x) * self.x[0]
        com_ref[1] = damping_factor_y * com_ref[1] + (1 - damping_factor_y) * self.x[1]
        com_ref[2] = self.conf.z0  # Keep height constant
        
        # Very conservative velocity and acceleration
        com_vel_ref = np.zeros(3)
        com_vel_ref[0] = (com_ref[0] - self.x[0]) * 0.5  # Very small gains
        com_vel_ref[1] = (com_ref[1] - self.x[1]) * 0.5
        com_vel_ref[2] = (com_ref[2] - self.x[2]) * 1.0
        
        com_acc_ref = np.zeros(3)
        com_acc_ref[0] = (com_vel_ref[0] - self.dx[0]) * 0.5
        com_acc_ref[1] = (com_vel_ref[1] - self.dx[1]) * 0.5
        com_acc_ref[2] = (com_vel_ref[2] - self.dx[2]) * 1.0
        
        # Rate limit the CoM reference to prevent any large jumps
        max_com_change = 0.001  # Very small rate limit
        if hasattr(self, 'prev_com_ref') and self.prev_com_ref is not None:
            com_change = com_ref - self.prev_com_ref
            # Apply rate limit to all components
            for i in range(3):
                if abs(com_change[i]) > max_com_change:
                    com_change[i] = np.sign(com_change[i]) * max_com_change
                    com_ref[i] = self.prev_com_ref[i] + com_change[i]
        
        self.prev_com_ref = com_ref.copy()
        
        com = np.array([com_ref, com_vel_ref, com_acc_ref])
        F_ext = np.zeros(3)
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
        # Only generate DCM trajectory at the start
        if not self.dcm_traj:
            self.gen_dcm_traj()
        
        # Debug prints to understand the issue
        print(f"DEBUG: current_step={self.current_step}, time_step={self.time_step}")
        print(f"DEBUG: len(dcm_traj)={len(self.dcm_traj)}, len(footstep_traj)={len(self.footstep_traj)}")
        
        # Get current DCM and ZMP references with bounds checking
        if (self.current_step < len(self.dcm_traj) and 
            self.time_step < len(self.dcm_traj[self.current_step]) and 
            self.current_step < len(self.footstep_traj)):
            
            print(f"DEBUG: Accessing dcm_traj[{self.current_step}][{self.time_step}]")
            dcm_ref = self.dcm_traj[self.current_step][self.time_step]
            
            # Get current robot state for VRP calculation
            data = self.biped.formulation.data()
            current_com = self.biped.robot.com(data)
            vrp_ref = current_com + np.array([0, 0, self.conf.z0])
            
            # Calculate DCM feedback control
            com_control, F_ext = self.dcm_controller(dcm_ref, vrp_ref)

            # Debug: Print CoM reference and actual CoM
            print(f"DEBUG: CoM reference: {com_control[0]}")
            print(f"DEBUG: Actual CoM: {self.x}")
            # Debug: Print foot references
            if self.current_step < len(self.footstep_traj):
                footstep = self.footstep_traj[self.current_step]
                if self.time_step < len(footstep[1]):
                    foot_transform = footstep[1][self.time_step]
                    print(f"DEBUG: Foot reference (current step): {foot_transform[:3, 3]}")
            # Debug: Print QP solver status (will be printed in main loop as well)

            # Set CoM task references using the correct TSID API
            self.biped.sample_com.value(com_control[0])
            self.biped.sample_com.derivative(com_control[1])
            self.biped.sample_com.second_derivative(com_control[2])
            # Use the vector directly instead of the TrajectorySample object
            self.biped.trajCom.setReference(com_control[0])
            self.logged_com_traj.append(com_control[0].copy())
            
            # Set foot position using the correct TSID pattern with safety checks
            try:
                footstep = self.footstep_traj[self.current_step]
                if footstep[0]: # Right foot
                    # Get the full transformation matrix from the footstep trajectory
                    if self.time_step < len(footstep[1]):
                        foot_transform = footstep[1][self.time_step]
                        
                        # Check if the foot position is reasonable
                        foot_pos = foot_transform[:3, 3]
                        current_foot_pos = self.biped.robot.framePosition(data, self.biped.RF).translation
                        foot_distance = np.linalg.norm(foot_pos - current_foot_pos)
                        
                        # If foot position is too far, use a closer position
                        max_foot_distance = 0.02  # More conservative maximum reasonable foot movement
                        if foot_distance > max_foot_distance:
                            print(f"Warning: Foot position too far ({foot_distance:.3f}m), limiting movement")
                            direction = (foot_pos - current_foot_pos) / foot_distance
                            foot_pos = current_foot_pos + direction * max_foot_distance
                            foot_transform[:3, 3] = foot_pos
                        
                        # Update the trajectory and compute next sample
                        self.biped.trajRF.setReference(pin.SE3(foot_transform))
                        self.biped.rightFootTask.setReference(self.biped.trajRF.computeNext())
                else: # Left foot
                    # Get the full transformation matrix from the footstep trajectory
                    if self.time_step < len(footstep[1]):
                        foot_transform = footstep[1][self.time_step]
                        
                        # Check if the foot position is reasonable
                        foot_pos = foot_transform[:3, 3]
                        current_foot_pos = self.biped.robot.framePosition(data, self.biped.LF).translation
                        foot_distance = np.linalg.norm(foot_pos - current_foot_pos)
                        
                        # If foot position is too far, use a closer position
                        max_foot_distance = 0.02  # More conservative maximum reasonable foot movement
                        if foot_distance > max_foot_distance:
                            print(f"Warning: Foot position too far ({foot_distance:.3f}m), limiting movement")
                            direction = (foot_pos - current_foot_pos) / foot_distance
                            foot_pos = current_foot_pos + direction * max_foot_distance
                            foot_transform[:3, 3] = foot_pos
                        
                        # Update the trajectory and compute next sample
                        self.biped.trajLF.setReference(pin.SE3(foot_transform))
                        self.biped.leftFootTask.setReference(self.biped.trajLF.computeNext())
            except Exception as e:
                print(f"Warning: Error setting foot trajectory: {e}")
                # Continue with current foot positions if trajectory setting fails
        else:
            print(f"DEBUG: Skipping update - bounds check failed")
            print(f"DEBUG: current_step={self.current_step}, time_step={self.time_step}")
            print(f"DEBUG: len(dcm_traj)={len(self.dcm_traj)}")
            if self.current_step < len(self.dcm_traj):
                print(f"DEBUG: len(dcm_traj[{self.current_step}])={len(self.dcm_traj[self.current_step])}")
            print(f"DEBUG: len(footstep_traj)={len(self.footstep_traj)}")

        # Update current step index - FIXED LOGIC
        self.time_step += 1
        if self.time_step >= int(self.conf.step_time / self.dt):
            self.time_step = 0
            self.current_step += 1
            print(f"DEBUG: Moving to next step: current_step={self.current_step}")
            if self.current_step >= len(self.footstep_traj):
                print("DEBUG: End of footstep trajectory reached")
                raise StopIteration("End of footstep trajectory")

        # Double support phase: only remove swing foot contact after 40% of the step time
        double_support_steps = int(0.4 * int(self.conf.step_time / self.dt))
        if self.current_step != self.last_step:
            self.swing_contact_removed = False
            self.last_step = self.current_step
        if (not getattr(self, 'swing_contact_removed', False)
            and self.time_step >= double_support_steps
            and self.current_step < len(self.footstep_traj)):
            footstep = self.footstep_traj[self.current_step]
            if footstep[0]:  # Right foot is swing
                print(f"[CONTACT] Removing right foot contact at step {self.current_step}")
                self.biped.removeRightFootContact()
                self.swing_foot = 'right'
            else:  # Left foot is swing
                print(f"[CONTACT] Removing left foot contact at step {self.current_step}")
                self.biped.removeLeftFootContact()
                self.swing_foot = 'left'
            self.swing_contact_removed = True
        # At the end of the step, add contact back for the swing foot
        # Add contact back earlier (at 80% of step time) for better stability
        if self.time_step >= int(0.8 * int(self.conf.step_time / self.dt)):
            if self.swing_foot == 'right' and not getattr(self, 'right_contact_added', False):
                print(f"[CONTACT] Adding right foot contact at step {self.current_step}")
                self.biped.addRightFootContact()
                self.right_contact_added = True
            elif self.swing_foot == 'left' and not getattr(self, 'left_contact_added', False):
                print(f"[CONTACT] Adding left foot contact at step {self.current_step}")
                self.biped.addLeftFootContact()
                self.left_contact_added = True
        # Reset contact flags at the start of each step
        if self.time_step == 0:
            self.right_contact_added = False
            self.left_contact_added = False

        return

    def reset(self):
        """Reset the controller state for a new trajectory."""
        self.time_step = 0
        self.current_step = 0
        self.footstep_traj = []
        self.dcm_traj = []
        self.com_traj = []
        self.logged_com_traj = []

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
            for t in traj:
                ax.plot(t[0, 3], t[1, 3], 'ro' if foot else 'bo')
        
        # Plot DCM trajectory
        for dcm in self.dcm_traj:
            for t in dcm:
                ax.plot(t[0], t[1], 'go')

        # Plot CoM trajectory
        for com in self.logged_com_traj:
            ax.plot(com[0], com[1], 'yo')

        # Plot ZMP trajectory
        for zmp in self.zmp_traj:
            for t in zmp:
                ax.plot(t[0], t[1], 'co')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Footstep and DCM Trajectory')
        ax.axis('equal')
        ax.grid()

    def get_support_foot(self):
        """Determine which foot is currently in support based on the walking phase.
        
        Returns:
            str: 'left', 'right', or 'both' indicating the support foot
        """
        if self.current_step >= len(self.footstep_traj):
            return 'both'  # Default to both feet if no trajectory
        
        footstep = self.footstep_traj[self.current_step]
        
        # During double support phase (first 40% of step), both feet are in contact
        double_support_steps = int(0.4 * int(self.conf.step_time / self.conf.dt))
        if self.time_step < double_support_steps:
            return 'both'
        
        # During single support phase, determine which foot is swing
        if footstep[0]:  # Right foot is swing
            return 'left'  # Left foot is support
        else:  # Left foot is swing
            return 'right'  # Right foot is support