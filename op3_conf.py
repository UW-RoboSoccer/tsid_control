import numpy as np

N_SIMULATION = 100000  # number of time steps simulated
dt = 0.002  # controller time step

step_length = 0.2  # length of the step
step_height = 0.05  # height of the step
step_width = 0.1275  # width of the step
step_time = 0.5  # time to complete the step

w_com = 1.0  # weight of center of mass task
w_am = 1e-3  # weight of angular momentum task
w_foot = 1e-1  # weight of the foot motion task
w_contact = -1.0  # weight of foot in contact (negative means infinite weight)
w_posture = 1e-1  # weight of joint posture task
w_forceRef = 1e-5  # weight of force regularization task
w_cop = 0.0
w_torque_bounds = 1e-1  # weight of the torque bounds
w_joint_bounds = 0.0

lyp = 0.055  # foot length in positive x direction
lyn = 0.055 # foot length in negative x direction
lxp = 0.0275  # foot length in positive y direction
lxn = 0.0275  # foot length in negative y direction
lz = 0.0  # foot sole height with respect to ankle joint
mu = 0.5  # friction coefficient
fMin = 0.0  # minimum normal force
fMax = 1000.0  # maximum normal force

tau_max_scaling = 3.0  # scaling factor of torque bounds
v_max_scaling = 10.0  # scaling factor of velocity bounds

kp_contact = 10.0  # proportional gain of contact constraint
kp_foot = 10.0  # proportional gain of contact constraint
kp_com = 10.0  # proportional gain of center of mass task
kp_am = 10.0  # proportional gain of angular momentum task
kp_posture = 1.0  # proportional gain of joint posture task

masks_posture = np.ones(18)

gain_vector = np.array([
    100.0, 100.0, # head
    10.0, 5.0, 5.0, 1.0, 1.0, # left leg
    10.0, 10.0, 10.0, # left arm
    10.0, 5.0, 5.0, 1.0, 1.0, # right leg
    10.0, 10.0, 10.0 # right arm
]) # gain vector for postural task

contactNormal = np.array(
    [0.0, 0.0, 1.0]
)  # direction of the normal to the contact surface

rf_frame_name = 'leg_right_sole_joint_fixed'
lf_frame_name = 'leg_left_sole_joint_fixed'

urdf = './robot/robot.urdf'
srdf = './robot/robot.srdf'
path_to_urdf = './robot'
mujoco_model_path = './robot/robot.xml'
