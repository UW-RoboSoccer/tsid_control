import numpy as np

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
        self.sim_time = conf.sim_time
        self.steps = int(self.sim_time / self.dt)
        self.current_step = 0
        
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
        self.footstep_traj = []  # Footstep trajectory
        self.zmp_traj = []       # ZMP trajectory
        self.dcm_traj = []       # DCM trajectory
        self.com_traj = []       # CoM trajectory
        
    def gen_footsteps(self, traj):
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
        right_foot = True  # Start with right foot

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
            
            # Generate cubic spline coefficients
            x_coefs = np.polyfit(t_spline, x_control, 3)
            y_coefs = np.polyfit(t_spline, y_control, 3)
            
            # Evaluate spline at each time point
            t_eval = np.linspace(0, 1, num_points)
            x_traj = np.polyval(x_coefs, t_eval)
            y_traj = np.polyval(y_coefs, t_eval)
            
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

            footstep_traj.append(HTM)

        self.footstep_traj = footstep_traj

    def gen_zmp_traj(self, depth=3):
        """Generate ZMP trajectory using cubic spline interpolation between footsteps.
        
        Creates a smooth ZMP trajectory for walking by interpolating between
        footstep positions with cubic splines.
        
        Args:
            depth: Number of future steps to consider
        """
        self.zmp_traj = []
        
        # We need at least two footsteps to create a trajectory
        if len(self.footstep_traj) < 2:
            print("Not enough footsteps to generate ZMP trajectory")
            return
        
        # Consider only the next 'depth' footsteps or all if fewer
        num_steps = min(len(self.footstep_traj) - self.current_step, depth)
        if num_steps < 2:
            print("Not enough future footsteps")
            return
        
        # For each step transition
        for i in range(num_steps - 1):
            step_idx = i + self.current_step
            if step_idx >= len(self.footstep_traj):
                break
                
            # Extract current and next footstep positions (initial point of each trajectory)
            p0 = self.footstep_traj[step_idx][0][:3, 3][:2]  # x,y of current footstep
            p1 = self.footstep_traj[step_idx + 1][0][:3, 3][:2]  # x,y of next footstep
            
            # Time points for cubic spline
            t_spline = np.array([0, 1])
            
            # Control points for the spline
            x_control = np.array([p0[0], p1[0]])
            y_control = np.array([p0[1], p1[1]])
            
            # Generate cubic spline coefficients
            x_coefs = np.polyfit(t_spline, x_control, 3)
            y_coefs = np.polyfit(t_spline, y_control, 3)
            
            # Evaluate spline at each time point
            num_points = int(self.conf.step_time / self.conf.dt)
            t_eval = np.linspace(0, 1, num_points)
            
            for j in range(num_points):
                # Calculate ZMP position using the spline
                zmp_x = np.polyval(x_coefs, t_eval[j])
                zmp_y = np.polyval(y_coefs, t_eval[j])
                
                # ZMP is typically on the ground, so z=0
                self.zmp_traj.append(np.array([zmp_x, zmp_y, 0.0]))
        
        # Add the final position explicitly
        final_step_idx = min(self.current_step + num_steps - 1, len(self.footstep_traj) - 1)
        final_pos = self.footstep_traj[final_step_idx][0][:3, 3]
        self.zmp_traj.append(np.array([final_pos[0], final_pos[1], 0.0]))

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
        vrp_pos = self.footstep_traj[depth-1][0][:3, 3]
        dcm_end = vrp_pos  # Terminal DCM equals the terminal VRP

        # Calculate DCM endpoints by backward recursion
        for i in range(depth-1, 0, -1):
            # Get VRP position (foot position)
            vrp_pos = self.footstep_traj[i + self.current_step][0][:3, 3]
            
            # Calculate initial DCM for this step to reach dcm_end at the end of the step
            dcm_i = self.back_calc_dcm(vrp_pos, dcm_end)
            
            # Update for next iteration
            dcm_end = dcm_i
            dcm_endpoints.insert(0, dcm_i)  # Insert at beginning to maintain order

        # Add final DCM endpoint
        dcm_endpoints.append(dcm_end)

        # Generate intermediate DCM points with exponential curves
        for i in range(depth-1):
            dcm_i = dcm_endpoints[i]      # Initial DCM for this step
            dcm_end = dcm_endpoints[i + 1]  # Final DCM for this step

            # Generate DCM points for this step
            for j in range(int(self.conf.step_time / self.dt)):
                t = j * self.dt
                
                # DCM evolution follows: ξ(t) = p + e^(t/T_c)*(ξ_0 - p)
                # where p is the VRP position and ξ_0 is the initial DCM
                vrp = self.footstep_traj[i + self.current_step][0][:3, 3]
                dcm_traj = vrp + (dcm_i - vrp) * np.exp(t / self.w_n)
                self.dcm_traj.append(dcm_traj)

        # Add the final DCM point
        self.dcm_traj.append(dcm_endpoints[-1])

    def gen_com_traj(self, x0, dx0):
        """Generate CoM trajectory from initial conditions and footsteps.
        
        Simulates the Center of Mass (CoM) trajectory using the Linear Inverted
        Pendulum Model (LIPM) dynamics and the planned footstep trajectory.
        
        Args:
            x0: Initial CoM position [x, y, z]
            dx0: Initial CoM velocity [dx, dy, dz]
        """
        # Generate CoM trajectory for each footstep and time step
        for i in range(len(self.footstep_traj)):
            for j in range(int(self.conf.step_time / self.dt)):
                # Linear Inverted Pendulum dynamics: ẍ = ω²(x - p)
                # where p is the ZMP position (assumed at footstep position)
                ddx = (1 / self.w_n**2) * (x0 - self.footstep_traj[i][0][:3, 3])
                
                # Integrate acceleration to get velocity
                dx = dx0 + ddx * self.dt
                
                # Integrate velocity to get position
                x = x0 + dx * self.dt
                
                # Update for next iteration
                x0 = x
                dx0 = dx
                
                # Store CoM state
                self.com_traj.append([ddx, dx, x])

    def back_calc_dcm(self, vrp_pos, dcm_end):
        """Calculate the initial DCM position to reach a desired DCM endpoint.
        
        Uses the exponential DCM dynamics to find the initial DCM position that will
        naturally evolve to the desired endpoint after a step duration.
        
        Args:
            vrp_pos: Virtual Repellent Point position [x, y, z]
            dcm_end: Desired DCM endpoint [x, y, z]
            
        Returns:
            Initial DCM position [x, y, z]
        """
        # DCM evolution is: ξ(t) = p + e^(t/T_c)*(ξ_0 - p)
        # Solving for ξ_0: ξ_0 = p + e^(-t/T_c)*(ξ(t) - p)
        dcm_ini = vrp_pos + (dcm_end - vrp_pos) * np.exp(-self.conf.step_time / self.w_n)
        return dcm_ini

    def dcm_controller(self, dcm_ref, d_dcm_ref):
        """DCM feedback controller to calculate ZMP command and external force.
        
        Implements a feedback controller for DCM tracking, calculating the required
        ZMP reference to drive the DCM toward the reference.
        
        Args:
            dcm_ref: Reference DCM position [x, y, z]
            d_dcm_ref: Reference DCM velocity [dx, dy, dz]
            
        Returns:
            zmp_control: Commanded ZMP position [x, y, z]
            F_ext: External force required [Fx, Fy, Fz]
        """
        # DCM feedback control law:
        # p = ξ + T_c*ξ̇ - T_c*k_ξ*(ξ - ξ_ref) - T_c*ξ̇_ref
        vrp_control = self.e + self.conf.e_gain * self.w_n * (self.e - dcm_ref) - self.w_n * d_dcm_ref
        
        # Convert VRP to ZMP by subtracting pendulum height in z direction
        zmp_control = vrp_control - np.array([0, 0, self.conf.z0])

        # Calculate required external force using the ZMP equation
        F_ext = (self.conf.m / (self.w_n**2)) * (self.x - zmp_control)

        return zmp_control, F_ext