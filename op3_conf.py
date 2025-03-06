import numpy as np

N_SIMULATION = 4000  # number of time steps simulated
dt = 0.002  # controller time step

w_com = 1.0  # weight of center of mass task
w_am = 1e-3  # weight of angular momentum task
w_foot = 1e-1  # weight of the foot motion task
w_contact = 1000000.0  # weight of foot in contact (negative means infinite weight)
w_posture = 1e-1  # weight of joint posture task
w_forceRef = 1e-5  # weight of force regularization task
w_cop = 0.0
w_torque_bounds = 0.0  # weight of the torque bounds
w_joint_bounds = 0.0

lxp = 0.055  # foot length in positive x direction
lxn = 0.055 # foot length in negative x direction
lyp = 0.0275  # foot length in positive y direction
lyn = 0.0275  # foot length in negative y direction
lz = 0.0  # foot sole height with respect to ankle joint
mu = 0.3  # friction coefficient
fMin = 0.0  # minimum normal force
fMax = 10000.0  # maximum normal force

tau_max_scaling = 3  # scaling factor of torque bounds
v_max_scaling = 1.0  # scaling factor of velocity bounds

kp_contact = 10.0  # proportional gain of contact constraint
kp_foot = 10.0  # proportional gain of contact constraint
kp_com = 10.0  # proportional gain of center of mass task
kp_am = 10.0  # proportional gain of angular momentum task
kp_posture = 10.0  # proportional gain of joint posture task

masks_posture = np.ones(18)

gain_vector = np.ones(18)  # gain vector for postural task

contactNormal = np.array(
    [0.0, 0.0, 1.0]
)  # direction of the normal to the contact surface

rf_frame_name = 'leg_right_sole_joint_fixed'
lf_frame_name = 'leg_left_sole_joint_fixed'

urdf = './robot/robot.urdf'
srdf = './robot/robot.srdf'
path_to_urdf = './robot'
mujoco_model_path = './robot/robot.xml'
