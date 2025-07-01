import numpy as np
import os

N_SIMULATION = 500  # number of time steps simulated
dt = 0.002  # controller time step
g = 9.81  # gravity

z0 = 0.4  # initial height of the center of mass

# Extremely conservative parameters for walking stability
step_length = 0.04  # Very short step length for stability (m)
step_height = 0.005  # Extremely low foot swing height (m)
step_width = 0.10   # Narrower width between feet (m)
step_time = 1.5     # Even longer time per step for stability (s)
linear_vel = 0.03   # Very slow linear velocity (m/s)

# DCM controller parameters - very conservative
k_dcm = 0.005  # very small DCM feedback gain
m = 1.5  # robot mass (kg)

# ULTRA-CONSERVATIVE: Make system much more stable than biped_balance.py
w_com = 1.0     # MATCH biped_balance.py exactly
w_am = 0.01     # MUCH LOWER: Reduce angular momentum influence
w_foot = 1e-5   # MINIMAL: Almost no foot motion tasks (like biped_balance.py)
w_contact = 100.0  # MATCH biped_balance.py exactly  
w_posture = 0.01   # MATCH biped_balance.py exactly
w_forceRef = 0.001 # MATCH biped_balance.py exactly
w_cop = 0.001  # MATCH biped_balance.py exactly
w_torque_bounds = 0.01  # LOWER: Conservative bounds
w_joint_bounds = 0.01   # LOWER: Conservative bounds

lyp = 0.055  # foot length in positive x direction
lyn = 0.055 # foot length in negative x direction
lxp = 0.0275  # foot length in positive y direction
lxn = 0.0275  # foot length in negative y direction
lz = 0.0  # foot sole height with respect to ankle joint
mu = 0.5  # friction coefficient
fMin = 0.0  # minimum normal force
fMax = 1000.0  # maximum normal force

tau_max_scaling = 1.0  # very conservative scaling factor of torque bounds
v_max_scaling = 2.0  # very conservative scaling factor of velocity bounds

# ULTRA-CONSERVATIVE: Much lower gains to prevent instability
kp_contact = 100.0  # KEEP HIGH: Contact needs to be stiff for ground contact
kp_foot = 1.0       # MUCH LOWER: Reduce foot tracking aggression  
kp_com = 1.0        # MUCH LOWER: Gentle CoM tracking to prevent oscillations
kp_am = 1.0         # MUCH LOWER: Gentle angular momentum control
kp_posture = 1.0    # REASONABLE: Keep joint posture moderate

masks_posture = np.ones(20)

# ULTRA-CONSERVATIVE gain vector - prevent all instability
gain_vector = np.array([
    0.1, 0.1, # head (2 joints) - minimal control
    0.05, 0.05, 0.05, 0.02, 0.02, # left leg (5 joints) - ULTRA conservative
    0.1, 0.1, 0.1, # left arm (3 joints) - minimal
    0.05, 0.05, 0.05, 0.02, 0.02, # right leg (5 joints) - ULTRA conservative  
    0.1, 0.1, 0.1, 0.05, 0.05 # right arm (5 joints) - minimal
]) # gain vector for postural task (20 joints total)

contactNormal = np.array(
    [0.0, 0.0, 1.0]
)  # direction of the normal to the contact surface

rf_frame_name = 'foot_sole'
lf_frame_name = 'foot_2_sole'

__dir__ = os.path.dirname(os.path.abspath(__file__))

urdf = os.path.join(__dir__, 'robot/v1/urdf/robot_mod.urdf')
srdf = os.path.join(__dir__, 'robot/v1/urdf/robot_mod.srdf')
path_to_urdf = os.path.join(__dir__, 'robot/v1')
mujoco_model_path = os.path.join(__dir__, 'robot/v1/mujoco/robot.xml')
