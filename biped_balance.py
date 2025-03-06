import mujoco.viewer
from biped import Biped
import op3_conf as conf

import mujoco

import time

import numpy as np

import pinocchio as pin

def map_tsid_to_mujoco(q_tsid):
    ctrl = np.zeros(18)
    ctrl[0] = q_tsid[22] # right shoulder pitch
    ctrl[1] = q_tsid[23] # right shoulder roll
    ctrl[2] = q_tsid[24] # right elbow

    ctrl[3] = q_tsid[14] # left shoulder pitch
    ctrl[4] = q_tsid[15] # left shoulder roll
    ctrl[5] = q_tsid[16] # left elbow

    ctrl[6] = q_tsid[7] # head yaw
    ctrl[7] = q_tsid[8] # head pitch

    ctrl[8] = q_tsid[17] # right hip pitch
    ctrl[9] = q_tsid[18] # right hip roll
    ctrl[10] = q_tsid[19] # right hip yaw
    ctrl[11] = q_tsid[20] # right knee
    ctrl[12] = q_tsid[21] # right ankle pitch

    ctrl[13] = q_tsid[9] # left hip pitch
    ctrl[14] = q_tsid[10] # left hip roll
    ctrl[15] = q_tsid[11] # left hip yaw
    ctrl[16] = q_tsid[12] # left knee
    ctrl[17] = q_tsid[13] # left ankle pitch

    return ctrl

biped = Biped(conf)

mj_model = mujoco.MjModel.from_xml_path(conf.mujoco_model_path)
mj_data = mujoco.MjData(mj_model)

push_robot_active, push_robot_com_vel, com_vel_entry = True, np.array([0.0, 0.0, 0.0]), None

com_0 = biped.robot.com(biped.formulation.data())

starttime = time.time()
amp = 0.05
freq = 0.25

# Set Mujoco timestep
mj_model.opt.timestep = conf.dt

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    viewer.sync()

    i, t = 0, 0.0
    q, v = biped.q, biped.v
    time_avg = 0.0

    HQPData = biped.formulation.computeProblemData(t, q, v)
    HQPData.print_all()

    mj_data.qpos = q

    print("Size of q: ", q.shape)
    print("Size of mujoco qpos: ", mj_data.ctrl.shape)

    print("\n=== PINOCCHIO/TSID JOINT STRUCTURE ===")
    pin_model = biped.model
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

    while viewer.is_running():
        # time_start = time.time()
        # t_elapsed = time.time() - starttime
        com_offset_x = amp * np.sin(2 * np.pi * freq * t)

        # Compute CoM reference and apply sinusoidal modification
        biped.trajCom.setReference(
            com_0 + np.array([0.0, com_offset_x, 0.0])
        )

        biped.comTask.setReference(biped.trajCom.computeNext())
        biped.postureTask.setReference(biped.trajPosture.computeNext())
        biped.rightFootTask.setReference(biped.trajRF.computeNext())
        biped.leftFootTask.setReference(biped.trajLF.computeNext())

        HQPData = biped.formulation.computeProblemData(t, q, v)

        print("com position: ", q[0:3])
        quit()
        print(mj_data.qpos[0:3])

        sol = biped.solver.solve(HQPData)
        if sol.status != 0:
            print("QP problem could not be solved! Error code:", sol.status)
            break

        # tau = biped.formulation.getActuatorForces(sol)
        dv = biped.formulation.getAccelerations(sol)
        print('a:', dv)
        q, v = biped.integrate_dv(q, v, dv, conf.dt)
        print('v:', v)
        print('q:', q)
        i, t = i + 1, t + conf.dt

        com = biped.robot.com(biped.formulation.data())
        print("CoM: ", com)
        print("CoM ref: ", biped.trajCom.getSample(t).value())

        # Add reference geom to follow com ref
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05, 0, 0],
            pos=biped.trajCom.getSample(t).value(),
            mat=np.eye(3).flatten(),
            rgba=np.array([1.0, 0.0, 0.0, 1.0]),
        )

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05, 0, 0],
            pos=com,
            mat=np.eye(3).flatten(),
            rgba=np.array([0.0, 1.0, 0.0, 1.0]),
        )

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[2],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05, 0, 0],
            pos=biped.robot.framePosition(biped.formulation.data(), biped.LF).translation,
            mat=np.eye(3).flatten(),
            rgba=np.array([0.0, 1.0, 1.0, 1.0]),
        )

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[3],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05, 0, 0],
            pos=biped.robot.framePosition(biped.formulation.data(), biped.RF).translation,
            mat=np.eye(3).flatten(),
            rgba=np.array([0.0, 0.0, 1.0, 1.0]),
        )

        viewer.user_scn.ngeom = 4

        mj_data.ctrl = map_tsid_to_mujoco(q)
        mujoco.mj_step(mj_model, mj_data)

        # current_time = time.time() - starttime

        # if current_time > 5.0:
        #     data = biped.formulation.data()
        #     push_robot_active = False
        #     b = push_robot_com_vel
        #     A = biped.comTask.compute(t, q, v, data).matrix
        #     v = np.linalg.lstsq(A, b, rcond=-1)[0]
        #     if current_time > 7.5:
        #         starttime = time.time()

        viewer.sync()

    while viewer.is_running():
        viewer.sync()
        time.sleep(1.0)