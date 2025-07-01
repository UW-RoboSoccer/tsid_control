import mujoco.viewer
from biped import Biped
import op3_conf as conf

import mujoco

import time
import csv

import numpy as np

def map_tsid_to_mujoco(q_tsid):
    ctrl = np.zeros(20)
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
    ctrl[18] = 0.0
    ctrl[19] = 0.0

    return ctrl

# Function to apply an external force to the robot's CoM
def apply_com_push(mj_model, mj_data, push_force):
    """Apply a force to the center of mass of the robot."""
    com_pos = mj_data.subtree_com[mj_model.body('torso').id]
    force = np.array(push_force)
    mj_data.xfrc_applied[mj_model.body('torso').id][:3] = force
    mj_data.xfrc_applied[mj_model.body('torso').id][3:] = np.cross(com_pos, force)

biped = Biped(conf)

# Initialize the MuJoCo model and data
mj_model = mujoco.MjModel.from_xml_path(conf.mujoco_model_path)
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = conf.dt

# Correctly initialize the robot's position in MuJoCo
q, v = biped.q, biped.v
mj_data.qpos = q
mj_data.qvel = v
mujoco.mj_step(mj_model, mj_data)
biped.formulation.computeProblemData(0.0, q, v)
left_foot_0 = biped.robot.framePosition(biped.formulation.data(), biped.LF).translation
right_foot_0 = biped.robot.framePosition(biped.formulation.data(), biped.RF).translation
feet_center = (left_foot_0 + right_foot_0) / 2.0
q[0] = feet_center[0]
q[1] = feet_center[1]
min_foot_height = min(left_foot_0[2], right_foot_0[2])
q[2] = q[2] - min_foot_height
mj_data.qpos = q
mj_data.qvel = v
mujoco.mj_step(mj_model, mj_data)
biped.formulation.computeProblemData(0.0, mj_data.qpos, mj_data.qvel)

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
# For logging
log_data = []
active_foot_traj = None
traj_start_time = 0

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        t_elapsed = time.time() - start_time

        # Compute CoM reference and apply sinusoidal modification
        com_offset_x = amp * np.sin(2 * np.pi * freq * t) # motion

        # Control
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
        is_falling = biped.falling()

        if is_falling and active_foot_traj is None:
            print(f"FALLING DETECTED at t={t:.2f}s! Generating recovery step.")
            active_foot_traj = biped.gen_footstep(cp, True, int(conf.step_time/conf.dt), conf.step_height)
            traj_start_time = i
            biped.removeRightFootContact()

        if active_foot_traj is not None:
            traj_idx = i - traj_start_time
            if traj_idx < len(active_foot_traj):
                biped.trajRF.setReference(active_foot_traj[traj_idx])
                biped.rightFootTask.setReference(biped.trajRF.computeNext())
            else:
                print(f"Recovery step finished at t={t:.2f}s. Re-enabling contact.")
                biped.addRightFootContact()
                active_foot_traj = None

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

        # Apply a push to the robot after 3 seconds
        push_active = t > 3.0 and t < 3.1
        if t > 3.0 and push_robot_active:
            print("Applying a push to the robot!")
            apply_com_push(mj_model, mj_data, [0, 200, 0]) # Push forward with 200N force
            push_robot_active = False

        # Reset the push force after a short duration
        if t > 3.1:
            apply_com_push(mj_model, mj_data, [0, 0, 0])

        # Log data for analysis
        log_data.append([
            t,
            biped.trajCom.getSample(t).value()[0], biped.trajCom.getSample(t).value()[1], biped.trajCom.getSample(t).value()[2],
            com[0], com[1], com[2],
            cp[0], cp[1],
            biped.robot.framePosition(biped.formulation.data(), biped.LF).translation[0], biped.robot.framePosition(biped.formulation.data(), biped.LF).translation[1],
            biped.robot.framePosition(biped.formulation.data(), biped.RF).translation[0], biped.robot.framePosition(biped.formulation.data(), biped.RF).translation[1],
            push_active,
            is_falling
        ])

        viewer.sync()

    # Save the log file
    with open('balance_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'com_ref_x', 'com_ref_y', 'com_ref_z', 'com_x', 'com_y', 'com_z', 'cp_x', 'cp_y', 'lfoot_x', 'lfoot_y', 'rfoot_x', 'rfoot_y', 'push_active', 'is_falling'])
        writer.writerows(log_data)

    while viewer.is_running():
        viewer.sync()
        time.sleep(1.0)

print("Simulation finished!")
