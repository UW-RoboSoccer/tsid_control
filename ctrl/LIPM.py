import numpy as np
import math
from Trajectory import Trajectory as Traj

class LIPM:
    def __init__(self, h0):
        '''
        pos [x, y]
        vel [vx, vy]
        acc [ax, ay]
        dcm [x, y]
        zmp [x, y]
        '''

        self.w = np.sqrt(9.80665 / h0)
        self.x = Traj()
        self.y = Traj()

    def pos(self, t):
        return np.array([self.x.get_frame(t, 0), self.y.get_frame(t, 0)])
    
    def vel(self, t):
        return np.array([self.x.get_frame(t, 1), self.y.get_frame(t, 1)])
    
    def acc(self, t):
        return np.array([self.x.get_frame(t, 2), self.y.get_frame(t, 2)])
    
    def dcm(self, t):
        return self.pos(t) + self.vel(t) / self.w
    
    def zmp(self, t):
        return self.pos(t) - self.acc(t) / self.w**2
    
    def make_trajectory(self, t, dt, pos0, vel0, acc0, zmp):
        '''
        pos [x, y]
        vel [vx, vy]
        acc [ax, ay]
        '''
        duration = t[1] - t[0]
        pos = pos0
        vel = vel0
        acc = acc0
        for i in range(math.floor(duration / dt)):
            acc = (zmp - pos) * self.w**2
            vel += acc * dt
            pos += vel * dt
            self.x.traj.append([pos[0], vel[0], acc[0]])
            self.y.traj.append([pos[1], vel[1], acc[1]])
