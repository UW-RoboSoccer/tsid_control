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

starttime = time.time()
amp = 0.5
freq = 0.1

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    viewer.sync()

    i, t = 0, 0.0
    q, v = biped.q, biped.v
    time_avg = 0.0

    HQPData = biped.formulation.computeProblemData(t, q, v)
    HQPData.print_all()

    while viewer.is_running():
        # time_start = time.time()
        t_elapsed = time.time() - starttime
        com_offset_x = amp * np.sin(2 * np.pi * freq * t_elapsed)

        # Compute CoM reference and apply sinusoidal modification
        sample_com = biped.trajCom.computeNext()
        com_ref = sample_com.value()
        com_ref[0] += com_offset_x  # Apply oscillation in x-direction
        sample_com.value(com_ref)
        print("CoM ref: ", com_ref)
        biped.comTask.setReference(sample_com)

        # Add reference geom to follow com ref
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05, 0, 0],
            pos=com_ref,
            mat=np.eye(3).flatten(),
            rgba=np.array([1.0, 0.0, 0.0, 1.0]),
        )

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05, 0, 0],
            pos=q[0:3],
            mat=np.eye(3).flatten(),
            rgba=np.array([0.0, 1.0, 0.0, 1.0]),
        )

        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[2],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.05, 0, 0],
            pos=mj_data.qpos[0:3],
            mat=np.eye(3).flatten(),
            rgba=np.array([0.0, 0.0, 1.0, 1.0]),
        )

        viewer.user_scn.ngeom = 3

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