import mujoco.viewer
from ctrl.WalkController import WalkController
from ctrl.conf import RobotConfig

import mujoco

import time

import numpy as np

def map_tsid_to_mujoco(q_tsid):
    ctrl = np.zeros(20)  # MuJoCo has 20 actuators (nu=20)
    
    # Right leg
    ctrl[0] = q_tsid[18]  # right_hip_yaw
    ctrl[1] = q_tsid[19]  # right_hip_roll
    ctrl[2] = q_tsid[20]  # right_hip_pitch
    ctrl[3] = q_tsid[21]  # right_knee
    ctrl[4] = q_tsid[22]  # right_ankle_pitch
    ctrl[5] = q_tsid[23]  # right_ankle_roll
    
    # Left leg
    ctrl[6] = q_tsid[9]   # left_hip_yaw
    ctrl[7] = q_tsid[10]  # left_hip_roll
    ctrl[8] = q_tsid[11]  # left_hip_pitch
    ctrl[9] = q_tsid[12]  # left_knee
    ctrl[10] = q_tsid[13] # left_ankle_pitch
    ctrl[11] = q_tsid[14] # left_ankle_roll
    
    # Left arm
    ctrl[12] = q_tsid[15] # left_shoulder_pitch
    ctrl[13] = q_tsid[16] # left_shoulder_roll
    ctrl[14] = q_tsid[17] # left_elbow
    
    # Right arm
    ctrl[15] = q_tsid[24] # right_shoulder_pitch
    ctrl[16] = q_tsid[25] # right_shoulder_roll
    ctrl[17] = q_tsid[26] # right_elbow
    
    # Head
    ctrl[18] = q_tsid[7]  # head_yaw
    ctrl[19] = q_tsid[8]  # head_pitch

    return ctrl

conf = RobotConfig()
controller = WalkController(conf)

# Initialize the MuJoCo model and data
mj_model = mujoco.MjModel.from_xml_path(conf.mjcf)
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = conf.dt

start_time = time.time()

# Initialize problem data
i, t = 0, 0.0
q, v = controller.q, controller.v

HQPData = controller.formulation.computeProblemData(t, q, v)
HQPData.print_all()

# Initialize Mujoco positions
mj_data.qpos = q

# Print joint structure for both Pinocchio and MuJoCo
print("Size of q: ", q.shape)
print("Size of mujoco qpos: ", mj_data.ctrl.shape)

print("\n=== PINOCCHIO/TSID JOINT STRUCTURE ===")
pin_model = controller.model
print(f"nq: {pin_model.nq}, nv: {pin_model.nv}, njoints: {pin_model.njoints}")
print("Joint ordering in Pinocchio:")
for i in range(1, pin_model.njoints):  # Skip the "universe" joint at index 0
    joint_name = pin_model.names[i]
    joint_id = pin_model.getJointId(joint_name)
    joint_idx = pin_model.idx_qs[joint_id] if hasattr(pin_model, 'idx_qs') else pin_model.joints[joint_id].idx_q
    print(f"Joint {i}: {joint_name}, q index: {joint_idx}")

print("\n=== MUJOCO JOINT STRUCTURE ===")
print(f"nq: {mj_model.nq}, nu: {mj_model.nu}")
print("Joint ordering in Mujoco:")
for i in range(mj_model.njnt):
    joint_name = mj_model.joint(i).name
    joint_type = mj_model.joint(i).type
    joint_qpos_addr = mj_model.joint(i).qposadr
    joint_actuator_idx = -1
    # Find corresponding actuator if any
    for j in range(mj_model.nu):
        if mj_model.actuator(j).trnid[0] == i:
            joint_actuator_idx = j
            break
    print(f"Joint {i}: {joint_name}, type: {joint_type}, qpos_addr: {joint_qpos_addr}, actuator_idx: {joint_actuator_idx}")

global running, paused
running = True
paused = False

def key_cb(key):
    global running, paused
    if chr(key) == 'q':
        print("Exiting simulation...")
        running = False

    if chr(key) == 'p':
        print("Pausing simulation...")
        paused = not paused

# Main simulation loop
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while running and viewer.is_running():
        if not paused:
            t_elapsed = time.time() - start_time

            # Compute CoM reference and apply sinusoidal modification
            # Control
            # controller.update_tasks(controller.traj_LF.computeNext(), controller.traj_RF.computeNext(), True, True)

            HQPData = controller.formulation.computeProblemData(t, q, v)

            sol = controller.solver.solve(HQPData)
            if sol.status != 0:
                print("QP problem could not be solved! Error code:", sol.status)
                break

            tau = controller.formulation.getActuatorForces(sol)
            dv = controller.formulation.getAccelerations(sol)
            q, v = controller.integrate_dv(q, v, dv, conf.dt)
            i, t = i + 1, t + conf.dt

            # Get CoP
            cop = controller.get_cop(sol)

            # Get CoM position
            com = controller.robot.com(controller.formulation.data())
            print(f"Time: {t:.2f}, CoM: {com}, CoP: {cop}")

            # Get frame positions
            LF_pos = controller.robot.framePosition(controller.formulation.data(), controller.LF_frame).translation
            LF_orientation = controller.robot.framePosition(controller.formulation.data(), controller.LF_frame).rotation
            RF_pos = controller.robot.framePosition(controller.formulation.data(), controller.RF_frame).translation
            RF_orientation = controller.robot.framePosition(controller.formulation.data(), controller.RF_frame).rotation
            print(f"LF position: {LF_pos}, RF position: {RF_pos}")
            print(f"LF orientation: {LF_orientation}, RF orientation: {RF_orientation}")

            # Visualize CoM
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[1],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.05, 0, 0],
                pos=com,
                mat=np.eye(3).flatten(),
                rgba=np.array([0.0, 1.0, 1.0, 0.75]),
            )

            # Visualize CoP
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0],
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.025, 0.0001, 0],
                pos=cop,
                mat=np.eye(3).flatten(),
                rgba=np.array([1.0, 0.0, 0.0, 1.0]),
            )

            # Visualize Contact points
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[2],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=[0.01, 0.02, 0.05],  # [shaft radius, head radius, head length]
                pos=controller.robot.framePosition(controller.formulation.data(), controller.LF_frame).translation,
                mat=LF_orientation.flatten(),
                rgba=np.array([0.0, 1.0, 0.0, 0.75]),
            )

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[3],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=[0.01, 0.02, 0.05],  # [shaft radius, head radius, head length]
                pos=controller.robot.framePosition(controller.formulation.data(), controller.RF_frame).translation,
                mat=RF_orientation.flatten(),
                rgba=np.array([0.0, 0.0, 1.0, 0.75]),
            )

            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD

            viewer.user_scn.ngeom = 4

            # ctrl = map_tsid_to_mujoco(q)
            # print("Control vector (MuJoCo):", ctrl)
            # mj_data.ctrl[:] = ctrl
            mj_data.qpos[:7] = q[:7]
            ctrl = map_tsid_to_mujoco(q)
            mj_data.ctrl = ctrl
            mujoco.mj_step(mj_model, mj_data)

            viewer.sync()
            controller.display(q)

        time.sleep(conf.dt / 2)

    while viewer.is_running():
        if not paused:
            viewer.sync()
            controller.display(q)
        time.sleep(conf.dt / 2)

print("Simulation finished!")
