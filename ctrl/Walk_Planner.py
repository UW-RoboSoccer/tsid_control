from typing import List

from ctrl.conf import *

from Foot_Trajectory import FootTrajectory
from Footstep_Planner import Footstep

"""
A class representing a trajectory through space and time.
"""
class Trajectory:
    pass

class WalkPlanner:
    def plan(self, footsteps: List[Footstep]):
        """
        Plan the walking trajectory based on the given footsteps.
        :param footsteps: List of Footstep objects representing the planned footsteps.
        :return: None
        """
        swing_trajectories = []

        for i in range(len(footsteps - 2)):
            start = [*footsteps[i].position, footsteps[i].orientation[2]]
            target = [*footsteps[i + 2].position, footsteps[i + 2].orientation[2]]
            foot_trajectory = FootTrajectory([0.0, step_duration],
                                             start=start,
                                             target=target,
                                             step_height=step_height,
                                             rise_ratio=rise_ratio)
            swing_trajectories.append(foot_trajectory)
            self.t += step_duration

        

