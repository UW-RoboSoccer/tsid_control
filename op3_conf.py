import numpy as np
import os

N_SIMULATION = 500  # number of time steps simulated
dt = 0.002  # controller time step
g = 9.81  # gravity

z0 = 0.4  # initial height of the center of mass

# Extremely conservative parameters for walking stability
step_length = 0.01  # very small step length (1cm)
step_height = 0.005  # very small height of the step (5mm)
step_width = 0.1275  # width of the step
step_time = 2.0  # longer time to complete the step

# DCM controller parameters - very conservative
k_dcm = 0.005  # very small DCM feedback gain
m = 1.5  # robot mass (kg)

# Very conservative task weights for walking stability
w_com = 1.0  # very small weight of center of mass task
w_am = 1e-2  # very small angular momentum task for balance
w_foot = 1e-5  # very tiny weight of the foot motion task
w_contact = 100.0  # very strong contact constraint
w_posture = 1e-2  # very small joint posture task for stability
w_forceRef = 1e-3  # very small weight of force regularization task
w_cop = 1e-3  # very small CoP control
w_torque_bounds = 1e-2  # very small weight of the torque bounds
w_joint_bounds = 1e-2  # very small joint bounds

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

# Very conservative gains for walking stability
kp_contact = 100.0  # very strong contact constraint
kp_foot = 1.0  # very small foot constraint
kp_com = 1.0  # very small center of mass task
kp_am = 1.0  # very small angular momentum task
kp_posture = 0.1  # very small joint posture task

masks_posture = np.ones(20)

# Very conservative gain vector for walking stability
gain_vector = np.array([
    0.5, 0.5, # head (2 joints) - very conservative
    0.2, 0.2, 0.2, 0.1, 0.1, # left leg (5 joints) - extremely conservative control
    0.2, 0.2, 0.2, # left arm (3 joints)
    0.2, 0.2, 0.2, 0.1, 0.1, # right leg (5 joints) - extremely conservative control
    0.2, 0.2, 0.2, 0.1, 0.1 # right arm (5 joints)
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
