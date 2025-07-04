import numpy as np
import mujoco
import mujoco.viewer
import time
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pinocchio as pin

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'ctrl'))

from ctrl.const import *
from ctrl.Footstep_Planner import FootstepPlanner, Footstep
from ctrl.Foot_Trajectory import FootTrajectory
from ctrl.Walk_Planner import WalkPlanner
from ctrl.LIPM import LIPM
from ctrl.CoMPlanner import CoMPlanner
from ctrl.DCM_Planner import DCMPlanner
from ctrl.ZMP_Planner import ZMPPlanner

import op3_conf as conf
from biped import Biped

class TSIDWalkingController:
    def __init__(self):
        self.biped = Biped(conf)