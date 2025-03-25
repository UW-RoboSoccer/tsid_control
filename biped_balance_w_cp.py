import mujoco.viewer
from biped import Biped
import op3_conf as conf
import capture_point

import mujoco

import time

import numpy as np

import pinocchio as pin

tsid_max_torque = 0.0
mujoco_max_torque = 0.0


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


def log_max_torque(ctrl, max_torque):
    for i in range(len(ctrl)):
        if abs(ctrl[i]) > abs(max_torque):
            max_torque = ctrl[i]
    return max_torque


biped = Biped(conf)

mj_model = mujoco.MjModel.from_xml_path(conf.mujoco_model_path)
mj_data = mujoco.MjData(mj_model)

push_robot_active, push_robot_com_vel, com_vel_entry = True, np.array([0.45, 0, 0.0]), None

com_0 = biped.robot.com(biped.formulation.data())

starttime = time.time()
amp = 0.5
freq = 0.1

tau_max = conf.tau_max
omega_nat = np.sqrt(conf.g/conf.z_com)
omega_f = 0

theta_max = 90 * np.pi / 180


with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    viewer.sync()

    i, t = 0, 0.0
    q, v = biped.q, biped.v
    time_avg = 0.0

    HQPData = biped.formulation.computeProblemData(t, q, v)
    HQPData.print_all()

    mj_data.qpos = q

    print("Joint state: ", q)

    while viewer.is_running():
        t_elapsed = time.time() - starttime

        # --- Capture Point Computation --- #
        x_dot_0 = v[0]
        omega_0 = 0
        theta_0 = 0
        capture_point_value = x_dot_0 * np.sqrt(com_0[2]/conf.g)
        # capture_point_value = capture_point.solve_capture_point(x_dot_0, tau_max, omega_0, theta_0, theta_max)
        print('Calculated Capture Point Value: ', capture_point_value)

        # --- update CoM Reference Trajectory and update comTask--- #
        sample_com = biped.trajCom.computeNext() 
        com_ref = sample_com.value()
        com_ref[0] += capture_point_value
        biped.trajCom.setReference(com_ref)

        # gain = 0.5
        # com_ref[0] = (1-gain) * com_ref[0] + gain * capture_point_value
        # sample_com.value(com_ref)
        biped.comTask.setReference(biped.trajCom.computeNext())
        # print("Updated CoM ref: ", com_ref)


        #Visualize the com_ref as a red sphere
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05, 0, 0],
            pos=com_ref,
            mat=np.eye(3).flatten(),
            rgba=np.array([1.0, 0.0, 0.0, 1.0]),
        )

        #Visualize the TSID COM position as a green sphere
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05, 0, 0],
            pos=q[0:3],
            mat=np.eye(3).flatten(),
            rgba=np.array([0.0, 1.0, 0.0, 1.0]),
        )

        #Visualize the mujoco COM position as blue sphere
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[2],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05, 0, 0],
            pos=mj_data.qpos[0:3],
            mat=np.eye(3).flatten(),
            rgba=np.array([0.0, 0.0, 1.0, 1.0]),
        )
        
        #Visualize the mujoco CP as a red circle on the x-y plane
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[3],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.025, 0.0001, 0],
            pos=[capture_point_value, com_0[1], 0],
            mat=np.eye(3).flatten(),
            rgba=np.array([1, 0, 0, 1.0]),
        )

        viewer.user_scn.ngeom = 4

        biped.postureTask.setReference(biped.trajPosture.computeNext())
        biped.rightFootTask.setReference(biped.trajRF.computeNext())
        biped.leftFootTask.setReference(biped.trajLF.computeNext())

        HQPData = biped.formulation.computeProblemData(t, q, v)

        sol = biped.solver.solve(HQPData)
        if sol.status != 0:
            print("QP problem could not be solved! Error code: ", sol.status)
            break

        dv = biped.formulation.getAccelerations(sol)
        q, v = biped.integrate_dv(q, v, dv, conf.dt)
        i, t = i + 1, t + conf.dt


        # --- Print the torques from TSID and MuJoCo for comparison --- #
        # tau = biped.formulation.getActuatorForces(sol)
        # print('TSID Torques:', tau)
        # print('Joint Torques: ', mj_data.qfrc_smooth + mj_data.qfrc_constraint)

        # --- Log the max torques observed to see if they violate the motor max torques --- #
        # tsid_max_torque = log_max_torque(tau, tsid_max_torque)
        # mujoco_max_torque = log_max_torque(mj_data.qfrc_smooth + mj_data.qfrc_constraint, mujoco_max_torque)
        # print('TSID max_torque obs: ', tsid_max_torque)
        # print('MuJoCo max_torque obs: ', mujoco_max_torque)

        # if (tsid_max_torque > 3):
        #     print('max torque passed, closing program')
        #     # quit()

        mj_data.ctrl = map_tsid_to_mujoco(q)
        mujoco.mj_step(mj_model, mj_data)

        current_time = time.time() - starttime

        if current_time > 5.0:
            data = biped.formulation.data()
            push_robot_active = False
            b = push_robot_com_vel
            A = biped.comTask.compute(t, q, v, data).matrix
            v = np.linalg.lstsq(A, b, rcond=-1)[0]
            if current_time > 5.25:
                starttime = time.time()

        viewer.sync()
    
    while viewer.is_running():
        viewer.sync()
