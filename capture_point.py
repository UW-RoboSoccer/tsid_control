import numpy as np
import matplotlib.pyplot as plt
import op3_conf as conf
import math

g = 9.81 #gravity factor
z_com = 0.22288998 #approx CoM height
m = 10 #Mass of the robot
#J = Flywheel inertia
tau_max = 2.914
# x_dot_0 = #Initial CoM velocity
omega_nat = np.sqrt(g/z_com) #LIPM natural frequency
# omega_0 = this value is based on te crrent value of the flywheel. not calculated
omega_f = 0 #need it to be 0 to solve for the capture region

def torque_profile(t, TR1, TR2):
    return (
        tau_max * (t >= 0)
        - 2 * tau_max * (t >= TR1)
        + tau_max * (t >= TR2)
    )

def solve_TR2( tau_min_max, J, omega_0 ):
    a = tau_min_max / (4 * J)
    b = 0.5 * (omega_f + omega_0)
    c = omega_0 - omega_max - J/(4*tau_min_max) * (omega_f - omega_0)**2

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("No real solution for TR2!")

    TR2_1 = (-b + np.sqrt(discriminant)) / (2**a)
    TR2_2 = (-b - np.sqrt(discriminant)) / (2**a)

    if TR2_1 > 0 and TR2_2 > 0:
        return min(TR2_1, TR2_2)
    elif TR2_1 < 0 and TR2_2 < 0:
        raise ValueError("No valid positive solution for TR2!")        
    else:
        return max(TR2_1, TR2_2)

def solve_TR1(TR2, J, omega_0):
    return (
        0.5 * TR2 + J / (2 * tau_max) * (omega_f - omega_0)
    )

def solve_capture_point(x_dot_0, tau_min_max, J, omega_0):
    TR2 = solve_TR2(tau_min_max)
    TR1 = solve_TR1(TR2, J, omega_0)
    return -1 * (
        (-1 / omega_nat)*x_dot_0 
        + (tau_max/(m*g)) * (math.exp(omega_nat*TR2) - 2*math.exp(omega_nat*(TR2-TR1))+1) / (math.exp(omega_nat*TR2))
    )

#this provides one of the bounds of the capture region

#To calcaulte the other boundary of the capture region, repeat with torque limit of tau_min and theta_min

#TODO:: Check if omega_not is 0, different calculation if it is

#get com_ref from biped.trajCom.computeNext().value()
#set com_ref[0] = capture_point
#biped.comTask.setReference(com_ref)

#tau_flywheel = torque_profile(t)
#flywheel_ref = np.array([0, 0, tau_flywheel])
# biped.flywheelTask.setReference(flywheel_ref)

# size of q is 25
# size of v and a is 24?
