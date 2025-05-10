import numpy as np

class Controller:
    def __init__(self, biped, conf):
        self.biped = biped
        self.conf = conf
        self.dt = conf.dt
        self.sim_time = conf.sim_time
        self.steps = int(self.sim_time / self.dt)
        self.current_step = 0
        
        # Initialize state variables
        self.x = np.zeros(3) # [x, y, z] CoM
        self.e = np.zeros(3) # [x, y, z] DCM
        self.state = np.zeros(2) # [x, e]
        self.zmp = np.zeros(2)
        self.footstep_traj = []
        self.dcm_traj = []
        
    def gen_footsteps(self, traj):
        self.current_step = 0
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

        self.footstep_traj = footstep_traj

    def gen_dcm_traj(self, depth=3):
        dcm_endpoints = []
        vrp_pos = self.footstep_traj[depth-1][0][:3, 3]
        dcm_end = vrp_pos

        for i in range(depth-1, 0, -1):
            vrp_pos = self.footstep_traj[i + self.current_step][0][:3, 3]
            dcm_i = self.back_calc_dcm(vrp_pos, dcm_end)
            dcm_end = dcm_i
            dcm_endpoints.insert(0, dcm_i)

        dcm_endpoints.append(dcm_end)

        for i in range(depth-1):
            dcm_i = dcm_endpoints[i]
            dcm_end = dcm_endpoints[i + 1]

            for j in range(int(self.conf.step_time / self.dt)):
                t = j * self.dt
                dcm_traj = self.footstep_traj[i + self.current_step][0][:3, 3] + (dcm_i - self.footstep_traj[i + self.current_step][0][:3, 3]) * np.exp(t / self.conf.tau)
                self.dcm_traj.append(dcm_traj)

        self.dcm_traj.append(dcm_endpoints[-1])

    def back_calc_dcm(self, vrp_pos, dcm_end):
        dcm_ini = np.zeros((3,))
        dcm_ini = vrp_pos + (dcm_end - vrp_pos) * np.exp(-self.conf.step_time / self.conf.tau)
        return dcm_ini
