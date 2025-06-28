import tsid
import pinocchio as pin

import numpy as np

class Biped:
    def __init__(self, conf):
        self.conf = conf

        self.robot = tsid.RobotWrapper(
            conf.urdf,
            [conf.path_to_urdf],
            pin.JointModelFreeFlyer(),
            False
        )

        robot = self.robot
        self.model = robot.model()

        pin.loadReferenceConfigurations(self.model, conf.srdf, False)
        
        # Try to load the standing configuration, but fall back to neutral if it fails
        try:
            self.q0 = q = self.model.referenceConfigurations["standing"]
        except KeyError:
            print("WARNING: Could not load 'standing' configuration from SRDF. Creating standing configuration.")
            # Create a standing configuration with feet on the ground
            self.q0 = q = pin.neutral(self.model)
            
            # Set the robot to stand with feet on the ground
            # The first 7 elements are [x, y, z, qw, qx, qy, qz] for the floating base
            # Set the height to be above the ground so feet touch
            q[2] = 0.4  # Set z height to place feet on ground
            
            # Adjust joint angles for a standing pose
            # These indices correspond to the joint ordering in the robot
            # You may need to adjust these based on your specific robot model
            q[7] = 0.0   # head yaw
            q[8] = 0.0   # head pitch
            
            # Left leg - standing pose with slight knee bend for stability
            q[9] = 0.0   # left hip pitch
            q[10] = 0.0  # left hip roll  
            q[11] = 0.0  # left hip yaw
            q[12] = 0.05  # left knee - smaller bend for more stability
            q[13] = -0.05 # left ankle pitch - compensate for knee bend
            
            # Right leg - standing pose with slight knee bend for stability
            q[17] = 0.0  # right hip pitch
            q[18] = 0.0  # right hip roll
            q[19] = 0.0  # right hip yaw
            q[20] = 0.05  # right knee - smaller bend for more stability
            q[21] = -0.05 # right ankle pitch - compensate for knee bend
            
            # Arms - neutral pose
            q[14] = 0.0  # left shoulder pitch
            q[15] = 0.0  # left shoulder roll
            q[16] = 0.0  # left elbow
            
            q[22] = 0.0  # right shoulder pitch
            q[23] = 0.0  # right shoulder roll
            q[24] = 0.0  # right elbow
        
        v = np.zeros(robot.nv)

        # Debug: Print available frames
        print("Available frames in robot model:")
        for i in range(self.model.nframes):
            frame_name = self.model.frames[i].name
            print(f"  {i}: {frame_name}")
        
        print(f"Looking for frames: {conf.rf_frame_name}, {conf.lf_frame_name}")
        
        assert self.model.existFrame(conf.rf_frame_name)
        assert self.model.existFrame(conf.lf_frame_name)

        formulation = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
        formulation.computeProblemData(0.0, q, v)
        data = formulation.data()

        contact_Point = np.ones((3, 4)) * (-conf.lz)
        contact_Point[0, :] = [-conf.lxn, -conf.lxn, conf.lxp, conf.lxp]
        contact_Point[1, :] = [-conf.lyn, conf.lyp, -conf.lyn, conf.lyp]

        contactRF = tsid.Contact6d(
            "contact_rfoot",
            robot,
            conf.rf_frame_name,
            contact_Point,
            conf.contactNormal,
            conf.mu,
            conf.fMin,
            conf.fMax,
        )
        contactRF.setKp(conf.kp_contact * np.ones(6))
        contactRF.setKd(2.0 * np.sqrt(conf.kp_contact) * np.ones(6))
        self.RF = robot.model().getFrameId(conf.rf_frame_name)
        H_rf_ref = robot.framePosition(data, self.RF)

        data = formulation.data()
        H_rf_ref = robot.framePosition(data, self.RF)
        contactRF.setReference(H_rf_ref)
        if conf.w_contact >= 0.0:
            formulation.addRigidContact(contactRF, conf.w_forceRef, conf.w_contact, 1)
        else:
            formulation.addRigidContact(contactRF, conf.w_forceRef)

        contactLF = tsid.Contact6d(
            "contact_lfoot",
            robot,
            conf.lf_frame_name,
            contact_Point,
            conf.contactNormal,
            conf.mu,
            conf.fMin,
            conf.fMax,
        )
        contactLF.setKp(conf.kp_contact * np.ones(6))
        contactLF.setKd(2.0 * np.sqrt(conf.kp_contact) * np.ones(6))
        self.LF = robot.model().getFrameId(conf.lf_frame_name)
        H_lf_ref = robot.framePosition(data, self.LF)
        print("Left foot position: ", H_lf_ref)
        contactLF.setReference(H_lf_ref)
        if conf.w_contact >= 0.0:
            formulation.addRigidContact(contactLF, conf.w_forceRef, conf.w_contact, 1)
        else:
            formulation.addRigidContact(contactLF, conf.w_forceRef)

        copTask = tsid.TaskCopEquality("task-cop", robot)
        formulation.addForceTask(copTask, conf.w_cop, 1, 0.0)

        amTask = tsid.TaskAMEquality("task-am", robot)
        amTask.setKp(conf.kp_am * np.array([1.0, 1.0, 0.0]))
        amTask.setKd(2.0 * np.sqrt(conf.kp_am * np.array([1.0, 1.0, 0.0])))
        formulation.addMotionTask(amTask, conf.w_am, 1, 0.0)
        sampleAM = tsid.TrajectorySample(3)
        amTask.setReference(sampleAM)

        comTask = tsid.TaskComEquality("task-com", robot)
        comTask.setKp(conf.kp_com * np.ones(3))
        comTask.setKd(2.0 * np.sqrt(conf.kp_com) * np.ones(3))
        formulation.addMotionTask(comTask, conf.w_com, 1, 0.0)

        postureTask = tsid.TaskJointPosture("task-posture", robot)
        print(f"DEBUG: gain_vector size: {len(conf.gain_vector)}")
        print(f"DEBUG: gain_vector: {conf.gain_vector}")
        print(f"DEBUG: robot.na (number of actuated joints): {robot.na}")
        postureTask.setKp(conf.kp_posture * conf.gain_vector)
        postureTask.setKd(2.0 * np.sqrt(conf.kp_posture * conf.gain_vector))
        postureTask.setMask(conf.masks_posture)
        formulation.addMotionTask(postureTask, conf.w_posture, 1, 0.0)

        self.leftFootTask = tsid.TaskSE3Equality(
            "task-left-foot", self.robot, self.conf.lf_frame_name
        )
        self.leftFootTask.setKp(self.conf.kp_foot * np.ones(6))
        self.leftFootTask.setKd(2.0 * np.sqrt(self.conf.kp_foot) * np.ones(6))
        self.trajLF = tsid.TrajectorySE3Constant("traj-left-foot", H_lf_ref)
        formulation.addMotionTask(self.leftFootTask, self.conf.w_foot, 1, 0.0)

        self.rightFootTask = tsid.TaskSE3Equality(
            "task-right-foot", self.robot, self.conf.rf_frame_name
        )
        self.rightFootTask.setKp(self.conf.kp_foot * np.ones(6))
        self.rightFootTask.setKd(2.0 * np.sqrt(self.conf.kp_foot) * np.ones(6))
        self.trajRF = tsid.TrajectorySE3Constant("traj-right-foot", H_rf_ref)
        formulation.addMotionTask(self.rightFootTask, self.conf.w_foot, 1, 0.0)

        self.tau_max = conf.tau_max_scaling * robot.model().effortLimit[-robot.na :]
        self.tau_min = -self.tau_max
        actuationBoundsTask = tsid.TaskActuationBounds("task-actuation-bounds", robot)
        actuationBoundsTask.setBounds(self.tau_min, self.tau_max)
        if conf.w_torque_bounds > 0.0:
            formulation.addActuationTask(
                actuationBoundsTask, conf.w_torque_bounds, 0, 0.0
            )

        jointBoundsTask = tsid.TaskJointBounds("task-joint-bounds", robot, conf.dt)
        self.v_max = conf.v_max_scaling * robot.model().velocityLimit[-robot.na :]
        self.v_min = -self.v_max
        jointBoundsTask.setVelocityBounds(self.v_min, self.v_max)
        if conf.w_joint_bounds > 0.0:
            formulation.addMotionTask(jointBoundsTask, conf.w_joint_bounds, 0, 0.0)

        com_ref = robot.com(data)
        self.trajCom = tsid.TrajectoryEuclidianConstant("traj_com", com_ref)
        self.sample_com = self.trajCom.computeNext()

        q_ref = q[7:]
        self.trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q_ref)
        postureTask.setReference(self.trajPosture.computeNext())

        self.sampleLF = self.trajLF.computeNext()
        self.sample_LF_pos = self.sampleLF.value()
        self.sample_LF_vel = self.sampleLF.derivative()
        self.sample_LF_acc = self.sampleLF.second_derivative()

        self.sampleRF = self.trajRF.computeNext()
        self.sample_RF_pos = self.sampleRF.value()
        self.sample_RF_vel = self.sampleRF.derivative()
        self.sample_RF_acc = self.sampleRF.second_derivative()

        self.solver = tsid.SolverHQuadProgFast("qp solver")
        self.solver.resize(formulation.nVar, formulation.nEq, formulation.nIn)

        self.comTask = comTask
        self.amTask = amTask
        self.copTask = copTask
        self.postureTask = postureTask
        self.contactRF = contactRF
        self.contactLF = contactLF
        self.actuationBoundsTask = actuationBoundsTask
        self.jointBoundsTask = jointBoundsTask
        self.formulation = formulation
        self.q = q
        self.v = v

        self.contact_LF_active = True
        self.contact_RF_active = True

    def removeLeftFootContact(self):
        if self.contact_LF_active:
            H_lf_ref = self.robot.framePosition(self.formulation.data(), self.LF)
            self.trajLF.setReference(H_lf_ref)
            self.leftFootTask.setReference(self.trajLF.computeNext())
            self.formulation.removeRigidContact(self.contactLF.name, 0.0)
            self.contact_LF_active = False

    def removeRightFootContact(self):
        if self.contact_RF_active:
            H_rf_ref = self.robot.framePosition(self.formulation.data(), self.RF)
            self.trajRF.setReference(H_rf_ref)
            self.rightFootTask.setReference(self.trajRF.computeNext())
            self.formulation.removeRigidContact(self.contactRF.name, 0.0)
            self.contact_RF_active = False

    def addLeftFootContact(self):
        if not self.contact_LF_active:
            H_lf_ref = self.robot.framePosition(self.formulation.data(), self.LF)
            self.contactLF.setReference(H_lf_ref)
            if self.conf.w_contact >= 0.0:
                self.formulation.addRigidContact(
                    self.contactLF, self.conf.w_forceRef, self.conf.w_contact, 1
                )
            else:
                self.formulation.addRigidContact(
                    self.contactLF, self.conf.w_forceRef
                )
            self.contact_LF_active = True

    def addRightFootContact(self):
        if not self.contact_RF_active:
            H_rf_ref = self.robot.framePosition(self.formulation.data(), self.RF)
            self.contactRF.setReference(H_rf_ref)
            if self.conf.w_contact >= 0.0:
                self.formulation.addRigidContact(
                    self.contactRF, self.conf.w_forceRef, self.conf.w_contact, 1
                )
            else:
                self.formulation.addRigidContact(
                    self.contactRF, self.conf.w_forceRef
                )
            self.contact_RF_active = True

    def gen_footstep(self, pos, r_foot, steps, height):
        pos0 = self.robot.framePosition(self.formulation.data(), self.RF if r_foot else self.LF).translation
        x = np.linspace(pos0[0], pos[0], steps)
        y = np.linspace(pos0[1], pos[1], steps)
        z = [4 * height * (i / steps) * (1 - i / steps) for i in range(steps)]
        traj = np.zeros((steps, 3))
        traj[:, 0] = x
        traj[:, 1] = y
        traj[:, 2] = z

    def compute_capture_point(self, com, dcom, w):
        cp = com + dcom / w
        cp[2] = 0
        return cp
    
    def compute_support_polygon(self):
        data = self.formulation.data()
        rf_pos = self.robot.framePosition(data, self.RF).translation
        lf_pos = self.robot.framePosition(data, self.LF).translation
        support_polygon = np.array([lf_pos[:2], rf_pos[:2]])
        return support_polygon

    def integrate_dv(self, q, v, dv, dt):
        v_mean = v + 0.5 * dt * dv
        v += dt * dv
        q = pin.integrate(self.model, q, dt * v_mean)
        return q, v