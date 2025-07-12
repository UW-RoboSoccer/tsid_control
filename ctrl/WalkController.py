from ctrl.conf import RobotConfig

import os
import time
import subprocess

import tsid
import numpy as np
import pinocchio as pin

class WalkController:
    def __init__(self, conf: RobotConfig):
        self.robot = tsid.RobotWrapper(
            conf.urdf,
            [conf.root_urdf],
            pin.JointModelFreeFlyer(),
            False
        )
        self.model = self.robot.model()
        self.formulation = tsid.InverseDynamicsFormulationAccForce("tsid", self.robot, False)

        pin.loadReferenceConfigurations(self.model, conf.srdf, False)
        self.q0 = self.q = self.model.referenceConfigurations["standing"]
        self.v = np.zeros(self.robot.nv)
        self.formulation.computeProblemData(0.0, self.q, self.v)
        self.data = self.formulation.data()

        if conf.visualizer:
            self.robot_display = pin.RobotWrapper.BuildFromURDF(
                conf.pin_urdf, [conf.root_urdf], pin.JointModelFreeFlyer()
            )

            launched = subprocess.getstatusoutput(
                    "ps aux |grep 'gepetto-gui'|grep -v 'grep'|wc -l"
            )
            if int(launched[1]) == 0:
                os.system("gepetto-gui &")
            time.sleep(1)
            self.viz = conf.visualizer(
                self.robot_display.model,
                self.robot_display.collision_model,
                self.robot_display.visual_model,
            )
            self.viz.initViewer(loadModel=True)
            self.viz.displayCollisions(False)
            self.viz.displayVisuals(True)
            self.viz.display(self.q)

            self.gui = self.viz.viewer.gui
            # self.gui.setCameraTransform(0, conf.CAMERA_TRANSFORM)
            self.gui.addFloor("world/floor")
            self.gui.setLightingMode("world/floor", "OFF")
        
        # Left foot contact task
        contact_points = np.ones((3, 4)) * (-conf.lz)
        contact_points[0, :] = [-conf.lxn, -conf.lxn, conf.lxp, conf.lxp]
        contact_points[1, :] = [-conf.lyn, conf.lyp, -conf.lyn, conf.lyp]

        contactLF = tsid.Contact6d(
            "contact_lfoot",
            self.robot,
            conf.lf_fixed_joint,
            contact_points,
            conf.contactNormal,  # Contact normal
            conf.mu,  # Friction coefficient
            conf.fMin,  # Min force
            conf.fMax  # Max force
        )
        contactLF.setKp(conf.kp_contact * np.ones(6))
        contactLF.setKd(2.0 * np.sqrt(conf.kp_contact) * np.ones(6))

        self.LF_frame = self.robot.model().getFrameId(conf.lf_fixed_joint)
        H_lf_ref = self.robot.framePosition(self.data, self.LF_frame)
        self.q[2] -= H_lf_ref.translation[2]  # Adjust the z-coordinate of the left foot

        self.formulation.computeProblemData(0.0, self.q, self.v)
        self.data = self.formulation.data()

        H_lf_ref = self.robot.framePosition(self.data, self.LF_frame)

        contactLF.setReference(H_lf_ref)
        if conf.w_contact >= 0.0:
            self.formulation.addRigidContact(contactLF, conf.w_forceRef, conf.w_contact, 1)
        else:
            self.formulation.addRigidContact(contactLF, conf.w_forceRef)
        self.contactLF = contactLF
        self.contactLF_active = True

        # Left foot trajectory task
        self.task_LF = tsid.TaskSE3Equality(
            "task_lfoot",
            self.robot,
            conf.lf_fixed_joint
        )
        self.task_LF.setKp(conf.kp_foot * np.ones(6))
        self.task_LF.setKd(2.0 * np.sqrt(conf.kp_foot) * np.ones(6))
        self.traj_LF = tsid.TrajectorySE3Constant("traj_lfoot", H_lf_ref)
        print("Left foot position: ", H_lf_ref)
        self.formulation.addMotionTask(
            self.task_LF,
            conf.w_foot,
            1,
            0.0
        )

        # Right foot contact task
        contactRF = tsid.Contact6d(
            "contact_rfoot",
            self.robot,
            conf.rf_fixed_joint,
            contact_points,
            conf.contactNormal,  # Contact normal
            conf.mu,  # Friction coefficient
            conf.fMin,  # Min force
            conf.fMax  # Max force
        )
        contactRF.setKp(conf.kp_contact * np.ones(6))
        contactRF.setKd(2.0 * np.sqrt(conf.kp_contact) * np.ones(6))
        self.RF_frame = self.robot.model().getFrameId(conf.rf_fixed_joint)
        H_rf_ref = self.robot.framePosition(self.data, self.RF_frame)
        print("Right foot position: ", H_rf_ref)
        contactRF.setReference(H_rf_ref)
        if conf.w_contact >= 0.0:
            self.formulation.addRigidContact(contactRF, conf.w_forceRef, conf.w_contact, 1)
        else:
            self.formulation.addRigidContact(contactRF, conf.w_forceRef)
        self.contactRF = contactRF
        self.contactRF_active = True

        # Right foot trajectory task
        self.task_RF = tsid.TaskSE3Equality(
            "task_rfoot",
            self.robot,
            conf.rf_fixed_joint
        )
        self.task_RF.setKp(conf.kp_foot * np.ones(6))
        self.task_RF.setKd(2.0 * np.sqrt(conf.kp_foot) * np.ones(6))
        self.traj_RF = tsid.TrajectorySE3Constant("traj_rfoot", H_rf_ref)
        self.formulation.addMotionTask(
            self.task_RF,
            conf.w_foot,
            1,
            0.0
        )

        # CoM task
        self.comTask = tsid.TaskComEquality("task-com", self.robot)
        self.comTask.setKp(conf.kp_com * np.ones(3))
        self.comTask.setKd(2.0 * np.sqrt(conf.kp_com) * np.ones(3))
        self.formulation.addMotionTask(self.comTask, conf.w_com, 1, 0.0)
        self.traj_COM = tsid.TrajectoryEuclidianConstant("traj-com", self.robot.com(self.data))
        self.comTask.setReference(self.traj_COM.computeNext())

        # CoP task

        # Orientation task

        # Posture task
        self.postureTask = tsid.TaskJointPosture("task-posture", self.robot)
        self.postureTask.setKp(conf.kp_posture * conf.gain_vector)
        self.postureTask.setKd(2.0 * np.sqrt(conf.kp_posture * conf.gain_vector))
        self.postureTask.setMask(conf.masks_posture)
        self.formulation.addMotionTask(self.postureTask, conf.w_posture, 1, 0.0)
        self.traj_posture = tsid.TrajectoryEuclidianConstant("traj-posture", self.q[7:])
        self.postureTask.setReference(self.traj_posture.computeNext())

        # Actuator Limits task
        self.tau_max = conf.tau_max_scaling * self.robot.model().effortLimit[-self.robot.na :]
        self.tau_min = -self.tau_max
        print("Torque limits:", self.tau_min, self.tau_max)
        self.actuationBoundsTask = tsid.TaskActuationBounds("task-actuation-bounds", self.robot)
        self.actuationBoundsTask.setBounds(self.tau_min, self.tau_max)
        if conf.w_torque_bounds > 0.0:
            self.formulation.addActuationTask(
                self.actuationBoundsTask, conf.w_torque_bounds, 0, 0.0
            )

        self.jointBoundsTask = tsid.TaskJointBounds("task-joint-bounds", self.robot, conf.dt)
        self.v_max = conf.v_max_scaling * self.robot.model().velocityLimit[-self.robot.na :]
        self.v_min = -self.v_max
        print("Velocity limits:", self.v_min, self.v_max)
        self.jointBoundsTask.setVelocityBounds(self.v_min, self.v_max)
        if conf.w_joint_bounds > 0.0:
            self.formulation.addMotionTask(self.jointBoundsTask, conf.w_joint_bounds, 0, 0.0)

        self.solver = tsid.SolverHQuadProgFast("qp solver")
        self.solver.resize(self.formulation.nVar, self.formulation.nEq, self.formulation.nIn)

    def update_tasks(self, sampleLF, sampleRF, contact_LF, contact_RF):
        """
        Update the tasks based on the current time.
        This method should be called at each control step.
        """
        # Update foot trajectories
        self.task_LF.setReference(sampleLF)
        self.task_RF.setReference(sampleRF)

        # Update contact tasks
        if contact_LF and not self.contactLF_active:
            self.add_contact(left_foot=True)
        elif not contact_LF and self.contactLF_active:
            self.remove_contact(left_foot=True)
        if contact_RF and not self.contactRF_active:
            self.add_contact(right_foot=True)
        elif not contact_RF and self.contactRF_active:
            self.remove_contact(right_foot=True)

        # Update CoM, CoP, and other tasks as needed
        # ...

    def display(self, q):
        if hasattr(self, 'viz'):
            self.viz.display(q)

    def remove_contact(self, left_foot=True, right_foot=True):
        """
        Remove contact tasks for the specified feet.
        """
        if left_foot and self.contactLF_active:
            T_lf = self.robot.framePosition(self.formulation.data(), self.LF_frame)
            self.traj_LF.setReference(T_lf)
            self.leftFootTask.setReference(self.traj_LF.computeNext())

            self.formulation.removeRigidContact(self.contactLF.name)
            self.contactLF_active = False
        if right_foot and self.contactRF_active:
            T_rf = self.robot.framePosition(self.formulation.data(), self.RF_frame)
            self.traj_RF.setReference(T_rf)
            self.rightFootTask.setReference(self.traj_RF.computeNext())

            self.formulation.removeRigidContact(self.contactRF)
            self.contactRF_active = False

    def add_contact(self, left_foot=True, right_foot=True):
        """
        Add contact tasks for the specified feet.
        """
        if left_foot and not self.contactLF_active:
            T_lf = self.robot.framePosition(self.formulation.data(), self.LF_frame)
            self.contactLF.setReference(T_lf)
            if self.conf.w_contact >= 0.0:
                self.formulation.addRigidContact(self.contactLF, self.conf.w_forceRef, self.conf.w_contact, 1)
            else:
                self.formulation.addRigidContact(self.contactLF, self.conf.w_forceRef)
            self.contactLF_active = True
        if right_foot and not self.contactRF_active:
            T_rf = self.robot.framePosition(self.formulation.data(), self.RF_frame)
            self.contactRF.setReference(T_rf)
            if self.conf.w_contact >= 0.0:
                self.formulation.addRigidContact(self.contactRF, self.conf.w_forceRef, self.conf.w_contact, 1)
            else:
                self.formulation.addRigidContact(self.contactRF, self.conf.w_forceRef)
            self.contactRF_active = True

    def get_cop(self, sol):
        """
        Get the center of pressure (CoP) based on the current contact forces.
        """
        cop_lf = np.zeros(3)
        cop_rf = np.zeros(3)
        if self.contactLF_active:
            T_lf_contact = self.contactLF.getForceGeneratorMatrix
            f_lf = T_lf_contact.dot(self.formulation.getContactForce(self.contactLF.name, sol))
            if f_lf[2] > 1e-3:
                cop_lf = np.array([
                    f_lf[4] / f_lf[2],
                    f_lf[3] / f_lf[2],
                    0.0
                ])

        if self.contactRF_active:
            T_rf_contact = self.contactRF.getForceGeneratorMatrix
            f_rf = T_rf_contact.dot(self.formulation.getContactForce(self.contactRF.name, sol))
            if f_rf[2] > 1e-3:
                cop_rf = np.array([
                    f_rf[4] / f_rf[2],
                    f_rf[3] / f_rf[2],
                    0.0
                ])
        
        cop_lf_world = self.robot.framePosition(self.data, self.LF_frame).act(cop_lf) if self.contactLF_active else None
        cop_rf_world = self.robot.framePosition(self.data, self.RF_frame).act(cop_rf) if self.contactRF_active else None

        cop = (cop_lf_world[:2] * f_lf[2] + cop_rf_world[:2] * f_rf[2]) / (f_lf[2] + f_rf[2]) if self.contactLF_active and self.contactRF_active else None
        # extend cop to 3D if it is not None
        if cop is not None:
            cop = np.append(cop, 0.0)

        return cop
    
    def integrate_dv(self, q, v, dv, dt):
        v_mean = v + 0.5 * dt * dv
        v += dt * dv
        q = pin.integrate(self.model, q, dt * v_mean)
        return q, v

