import numpy as np
import matplotlib.pyplot as plt
import op3_conf as conf
import math
from biped import Biped

biped = Biped(conf)

g = conf.g #gravity factor
z_com = conf.z_com #approx CoM height
m = 10 #Mass of the robot
tau_max = conf.tau_max
omega_nat = np.sqrt(g/z_com) #LIPM natural frequency
omega_f = 0 #need it to be 0 to solve for the capture region

torso_id = biped.model.getBodyId("torso")
torso_inertial = biped.model.inertias[torso_id]
J = torso_inertial.inertia[1,1] #Set flywheel inertia as Jyy from the inertia matrix

def torque_profile(t, TR1, TR2):
    return (
        tau_max * (t >= 0)
        - 2 * tau_max * (t >= TR1)
        + tau_max * (t >= TR2)
    )

def solve_TR( tau_min_max, omega_0, theta_0, theta_max ):

    if omega_0 == 0:
        TR2 = math.sqrt(4 * J * (theta_max - theta_0) / tau_min_max )
        TR1 = 0.5 * TR2
        return (TR1, TR2)
    
    else:
        a = tau_min_max / (4 * J)
        b = 0.5 * (omega_f + omega_0)
        c = theta_0 - theta_max - J/(4*tau_min_max) * (omega_f - omega_0)**2

        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            raise ValueError("No real solution for TR2!")

        TR2_1 = (-b + np.sqrt(discriminant)) / (2**a)
        TR2_2 = (-b - np.sqrt(discriminant)) / (2**a)

        if TR2_1 > 0 and TR2_2 > 0:
            TR2 = min(TR2_1, TR2_2)
        elif TR2_1 < 0 and TR2_2 < 0:
            raise ValueError("No valid positive solution for TR2!")        
        else:
            TR2 = max(TR2_1, TR2_2)
        
        TR1 =  (0.5 * TR2) + (J / (2 * tau_max)) * (omega_f - omega_0)

        return (TR1, TR2)

def solve_capture_point(x_dot_0, tau_min_max, omega_0, theta_0, theta_max):
    TR1, TR2 = solve_TR(tau_min_max, omega_0, theta_0, theta_max)

    return -1 * (
        (-1 / omega_nat)*x_dot_0 
        + (tau_max/(m*g)) * (math.exp(omega_nat*TR2) - 2*math.exp(omega_nat*(TR2-TR1))+1) / (math.exp(omega_nat*TR2))
    )

#this provides one of the bounds of the capture region

#To calcaulte the other boundary of the capture region, repeat with torque limit of tau_min and theta_min
