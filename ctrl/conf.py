import numpy as np
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer

class RobotConfig:
    """Configuration parameters for the robot controller."""
    
    # Robot configuration
    robot_path = "./robot/v1"
    
    root_urdf = f"{robot_path}/urdf"
    urdf = f"{robot_path}/urdf/robot_mod.urdf"
    pin_urdf = f"{root_urdf}/robot_mod.urdf"
    mjcf = f"{robot_path}/mujoco/scene.xml"
    srdf = f"{root_urdf}/robot.srdf"

    lf_fixed_joint = "left_sole_joint_fixed"
    rf_fixed_joint = "right_sole_joint_fixed"
    
    # Controller settings
    dt = 0.002  # controller time step
    
    # Step parameters
    step_height = 0.2  # height of the step
    step_width = 0.2  # width of the step
    step_length = 0.3  # length of the step
    step_duration = 0.5  # duration of a step
    rise_ratio = 0.5  # ratio of the step duration for rising and falling
    
    # Frame dimensions
    lxn = 0.055  # length of the x-axis in the world frame
    lyn = 0.0275  # length of the y-axis in the world frame
    lxp = 0.055  # length of the x-axis in the robot frame
    lyp = 0.0275  # length of the y-axis in the robot frame
    lz = 0.0  # height of the contact point in the z-axis

    # Contact parameters
    mu = 0.5  # friction coefficient
    fMin = 10.0  # minimum normal force
    fMax = 1000.0  # maximum normal force
    contactNormal = np.array([0.0, 0.0, 1.0])  # direction of the normal to the contact surface
    w_contact = -1.0  # weight of foot in contact (negative means infinite weight)
    w_forceRef = 1e-5  # weight of force regularization task
    kp_contact = 10.0  # proportional gain of contact constraint

    # Foot trajectory parameters
    w_foot = 1e-1  # weight of the foot motion task
    kp_foot = 10.0  # proportional gain of foot trajectory task

    # CoM parameters
    w_com = 1e-1
    kp_com = 10.0  # proportional gain of CoM task

    # Posture parameters
    w_posture = 1e-1
    kp_posture = 10.0
    gain_vector = np.array([
        100.0, 100.0, # head
        10.0, 5.0, 5.0, 1.0, 1.0, 1.0, # left leg
        10.0, 10.0, 10.0, # left arm
        10.0, 5.0, 5.0, 1.0, 1.0, 1.0, # right leg
        10.0, 10.0, 10.0 # right arm
    ]) # gain vector for postural task

    # Masks for posture task
    masks_posture = np.ones(20)  # mask for the posture task, 20 joints

    # Joint limit parameters
    tau_max_scaling = 5.0  # maximum torque for the joints
    v_max_scaling = 10.0  # maximum velocity for the joints
    w_torque_bounds = 1e-2  # weight of the torque bounds task
    w_joint_bounds = 1e-2  # weight of the joint limits task

    ### WARNING: Don't set, this is currently not working
    visualizer = None # Visualizer class, e.g., GepettoVisualizer