import mujoco.viewer
from biped import Biped
import op3_conf as conf

import mujoco

import time

import numpy as np

import pinocchio as pin

biped = Biped(conf)

mj_model = mujoco.MjModel.from_xml_path(conf.mujoco_model_path)
mj_data = mujoco.MjData(mj_model)

push_robot_active, push_robot_com_vel, com_vel_entry = True, np.array([0.0, 0.0, 0.0]), None

com_0 = biped.robot.com(biped.formulation.data())

starttime = time.time()
amp = 0.1
freq = 0.1

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

        sol = biped.solver.solve(HQPData)
        if sol.status != 0:
            print("QP problem could not be solved! Error code:", sol.status)
            break

        # tau = biped.formulation.getActuatorForces(sol)
        dv = biped.formulation.getAccelerations(sol)
        q, v = biped.integrate_dv(q, v, dv, conf.dt)
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
            pos=biped.trajLF.getSample(t).value()[:3],
            mat=np.eye(3).flatten(),
            rgba=np.array([0.0, 0.0, 1.0, 1.0]),
        )

        viewer.user_scn.ngeom = 3

        ctrl = q[7:]
        mj_data.ctrl = ctrl
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