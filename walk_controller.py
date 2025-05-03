import numpy as np

class Controller:
    def __init__(self, biped, conf):
        self.biped = biped
        self.conf = conf
        self.dt = conf.dt
        self.sim_time = conf.sim_time
        self.steps = int(self.sim_time / self.dt)
        
        # Initialize state variables
        self.state = np.zeros(4) # [x, y, dx, dy]
        self.zmp = np.zeros(2)
        self.capture_point = np.zeros(2)
        self.support_polygon = np.zeros((2, 2))
        self.footstep_traj = []

    def set_state(self, position, velocity):
        self.state[0] = position[0]
        self.state[1] = position[1]
        self.state[2] = velocity[0]
        self.state[3] = velocity[1]

    def compute_capture_point(self):
        # ξ = x + ẋ/ω
        omega = np.sqrt(self.conf.g / self.conf.h)
        self.capture_point = self.state[0:2] + self.state[2:4] / omega
        return self.capture_point
        
    def gen_footsteps(self, traj):
        footsteps = []
        dist = 0
        right_foot = True

        for i in range(len(traj) - 1):
            dx = traj[i + 1, 0] - traj[i, 0]
            dy = traj[i + 1, 1] - traj[i, 1]

            dist += np.sqrt(dx ** 2 + dy ** 2)
            if dist >= self.conf.step_length:
                tangent = [dx, dy]
                tangent /= np.linalg.norm(tangent)
                normal = [-tangent[1], tangent[0]]
                
                footstep = np.zeros((4,))
                footstep[0] = traj[i, 0] + self.conf.step_width * normal[0] * (1 if right_foot else -1)
                footstep[1] = traj[i, 1] + self.conf.step_width * normal[1] * (1 if right_foot else -1)
                footstep[2] = np.arctan2(tangent[1], tangent[0])
                footstep[3] = right_foot
                footsteps.append(footstep)
                dist = 0
                right_foot = not right_foot
    
        footstep_traj = []
        num_points = int(self.conf.step_time / self.conf.dt)
        t = np.linspace(0, 1, num_points)
        for i in range(len(footsteps) - 2):
            p0 = footsteps[i][:2]
            p1 = footsteps[i + 2][:2]

            x_traj = np.linspace(p0[0], p1[0], num_points)
            y_traj = np.linspace(p0[1], p1[1], num_points)
            z_traj = 4 * self.conf.step_height * t * (1 - t)

            HTM = []
            yaw = footsteps[i + 1][2]
            for j in range(num_points):
                T = np.eye(4)
                R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])
                T[:3, :3] = R
                T[:3, 3] = [x_traj[j], y_traj[j], z_traj[j]]
                HTM.append(T)

            footstep_traj.append(HTM)