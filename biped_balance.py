import mujoco.viewer
from biped import Biped
import op3_conf as conf

import mujoco

import time

import numpy as np

biped = Biped(conf)

mj_model = mujoco.MjModel.from_xml_path(conf.mujoco_model_path)
mj_data = mujoco.MjData(mj_model)

push_robot_active, push_robot_com_vel, com_vel_entry = True, np.array([0.0, 0.0, 0.0]), None

starttime = time.time()


with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    viewer.sync()

    i, t = 0, 0.0
    q, v = biped.q, biped.v
    time_avg = 0.0


    while viewer.is_running():
        time_start = time.time()

        biped.comTask.setReference(biped.trajCom.computeNext())
        biped.postureTask.setReference(biped.trajPosture.computeNext())
        biped.rightFootTask.setReference(biped.trajRF.computeNext())
        biped.leftFootTask.setReference(biped.trajLF.computeNext())

        HQPData = biped.formulation.computeProblemData(t, q, v)

        print("com position: ", q[0:3])

        sol = biped.solver.solve(HQPData)
        if sol.status != 0:
            print("QP problem could not be solved! Error code:", sol.status)
            break

        # tau = biped.formulation.getActuatorForces(sol)
        dv = biped.formulation.getAccelerations(sol)
        q, v = biped.integrate_dv(q, v, dv, conf.dt)
        i, t = i + 1, t + conf.dt

        print("com position: ", q[0:3])

        ctrl = q[7:]
        mj_data.ctrl = ctrl
        mujoco.mj_step(mj_model, mj_data)

        current_time = time.time() - starttime

        if current_time > 5.0:
            data = biped.formulation.data()
            push_robot_active = False
            b = push_robot_com_vel
            A = biped.comTask.compute(t, q, v, data).matrix
            v = np.linalg.lstsq(A, b, rcond=-1)[0]
            if current_time > 7.5:
                starttime = time.time()

        viewer.sync()