from ctrl.conf import dt
import math

class Trajectory:
    def __init__(self):
        self.traj = []

    def get_frame(self, t, diff):
        k = math.floor(t / dt)

        if k < 0 or k >= len(self.traj):
            raise IndexError("Time index out of bounds")
        if diff < 0 or diff >= len(self.traj[t]):
            raise IndexError("Difference index out of bounds")
        return self.traj[k][diff]