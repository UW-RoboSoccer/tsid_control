import mujoco.viewer
from biped import Biped
import op3_conf as conf

import mujoco

import time

import numpy as np

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

# Initialize the MuJoCo model and data
mj_model = mujoco.MjModel.from_xml_path(conf.mujoco_model_path)
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = conf.dt

push_robot_active, push_robot_com_vel, com_vel_entry = True, np.array([0.0, -0.1, 0.0]), None

start_time = time.time()
amp = 0.0
freq = 0.0

# Initialize problem data
i, t = 0, 0.0
q, v = biped.q, biped.v
com_0 = biped.robot.com(biped.formulation.data())

HQPData = biped.formulation.computeProblemData(t, q, v)
HQPData.print_all()

# Initialize Mujoco positions
mj_data.qpos = q

# Print joint structure for both Pinocchio and MuJoCo
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

# Main simulation loop
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        t_elapsed = time.time() - start_time

        # Compute CoM reference and apply sinusoidal modification
        com_offset_x = amp * np.sin(2 * np.pi * freq * t)
        biped.trajCom.setReference(
            com_0 + np.array([0.0, com_offset_x, 0.0])
        )

        biped.comTask.setReference(biped.trajCom.computeNext())
        biped.postureTask.setReference(biped.trajPosture.computeNext())
        biped.rightFootTask.setReference(biped.trajRF.computeNext())
        biped.leftFootTask.setReference(biped.trajLF.computeNext())

        HQPData = biped.formulation.computeProblemData(t, q, v)

        sol = biped.solver.solve(HQPData)
        if sol.status != 0:
            print("QP problem could not be solved! Error code:", sol.status)
            break

        tau = biped.formulation.getActuatorForces(sol)
        dv = biped.formulation.getAccelerations(sol)
        q, v = biped.integrate_dv(q, v, dv, conf.dt)
        i, t = i + 1, t + conf.dt

        # Get center of mass velocity
        com = biped.robot.com(biped.formulation.data())
        com_vel = biped.robot.com_vel(biped.formulation.data())
        w = np.sqrt(9.81 / com_0[2])
        cp = biped.compute_capture_point(com, com_vel, w)

        if biped.falling():
            start = i
            footstep = biped.gen_footstep(cp, True, conf.step_time/conf.dt, conf.step_height)
            biped.removeRightFootContact()

            biped.trajRF.setReference(footstep[start - i])

        # Add reference geom to follow com ref
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05, 0, 0],
            pos=biped.trajCom.getSample(t).value(),
            mat=np.eye(3).flatten(),
            rgba=np.array([1.0, 0.0, 0.0, 1.0]),
        )

        # Add reference geom to follow com
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05, 0, 0],
            pos=com,
            mat=np.eye(3).flatten(),
            rgba=np.array([0.0, 1.0, 0.0, 1.0]),
        )

        # Add reference geom to follow contact points
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[2],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.025, 0, 0],
            pos=biped.robot.framePosition(biped.formulation.data(), biped.LF).translation,
            mat=np.eye(3).flatten(),
            rgba=np.array([0.0, 1.0, 1.0, 0.5]),
        )

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[3],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.025, 0, 0],
            pos=biped.robot.framePosition(biped.formulation.data(), biped.RF).translation,
            mat=np.eye(3).flatten(),
            rgba=np.array([0.0, 0.0, 1.0, 0.5]),
        )

        # Add capture point geom
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[4],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.025, 0.0001, 0],
            pos=cp,
            mat=np.eye(3).flatten(),
            rgba=np.array([1.0, 1.0, 0.0, 1.0]),
        )
        
        viewer.user_scn.ngeom = 5

        mj_data.ctrl = map_tsid_to_mujoco(q)
        mujoco.mj_step(mj_model, mj_data)

        if t_elapsed > 5.0:
            print("Pushing robot")
            push_robot_active = False
            data = biped.formulation.data()
            J_LF = biped.contactLF.computeMotionTask(0.0, q, v, data).matrix
            J_RF = biped.contactRF.computeMotionTask(0.0, q, v, data).matrix
            J = np.vstack((J_LF, J_RF))
            J_com = biped.comTask.compute(t, q, v, data).matrix
            A = np.vstack((J_com, J))
            b = np.concatenate((np.array(push_robot_com_vel), np.zeros(J.shape[0])))
            v = np.linalg.lstsq(A, b, rcond=-1)[0]
            starttime = time.time()

        viewer.sync()

    while viewer.is_running():
        viewer.sync()
        time.sleep(1.0)

print("Simulation finished!")
