import mujoco.viewer
from biped import Biped
import op3_conf as conf
import pinocchio as pin
import mujoco
import time
import numpy as np
import sys
import os

# Add ctrl directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ctrl'))
from Footstep_Planner import FootstepPlanner, Footstep, Support
from LIPM import LIPM
from Foot_Trajectory import FootTrajectory
from const import *

def map_tsid_to_mujoco(q_tsid):
    ctrl = np.zeros(20)
    
    # Right leg (actuators 0-5)
    ctrl[0] = q_tsid[18] # right hip yaw
    ctrl[1] = q_tsid[19] # right hip roll
    ctrl[2] = q_tsid[20] # right hip pitch
    ctrl[3] = q_tsid[21] # right knee
    ctrl[4] = q_tsid[22] # right ankle pitch
    ctrl[5] = q_tsid[23] # right ankle roll

    # Left leg (actuators 6-11)
    ctrl[6] = q_tsid[9]  # left hip yaw
    ctrl[7] = q_tsid[10] # left hip roll
    ctrl[8] = q_tsid[11] # left hip pitch
    ctrl[9] = q_tsid[12] # left knee
    ctrl[10] = q_tsid[13] # left ankle pitch
    ctrl[11] = q_tsid[14] # left ankle roll

    # Left arm (actuators 12-14)
    ctrl[12] = q_tsid[15] # left shoulder pitch
    ctrl[13] = q_tsid[16] # left shoulder roll
    ctrl[14] = q_tsid[17] # left elbow

    # Right arm (actuators 15-17)
    ctrl[15] = q_tsid[24] # right shoulder pitch
    ctrl[16] = q_tsid[25] # right shoulder roll
    ctrl[17] = q_tsid[26] # right elbow

    # Head (actuators 18-19)
    ctrl[18] = q_tsid[7]  # head yaw
    ctrl[19] = q_tsid[8]  # head pitch

    return ctrl

class EnhancedBalanceController:
    def __init__(self, biped, conf):
        self.biped = biped
        self.conf = conf
        self.dt = conf.dt
        
        # Initialize footstep planner
        self.footstep_planner = FootstepPlanner(step_width=step_width, step_length=step_length)
        
        # Initialize LIPM for CoM trajectory
        self.lipm = LIPM(conf.z0)
        
        # Current support state
        self.current_support = None
        self.footsteps = []
        self.current_footstep_idx = 0
        self.step_phase = "double_support"  # "double_support", "single_support", "stepping"
        self.step_start_time = 0.0
        self.step_duration = step_duration
        self.recovery_in_progress = False
        
        # Initialize footsteps based on current robot state
        self.initialize_footsteps()
        
    def initialize_footsteps(self):
        """Initialize footsteps based on current robot foot positions"""
        data = self.biped.formulation.data()
        lf_pos = self.biped.robot.framePosition(data, self.biped.LF).translation
        rf_pos = self.biped.robot.framePosition(data, self.biped.RF).translation
        
        # Create initial footsteps
        left_footstep = Footstep(
            position=lf_pos[:2],  # Only x,y
            orientation=np.array([0, 0, 0]),  # No rotation initially
            side=0  # Left foot
        )
        
        right_footstep = Footstep(
            position=rf_pos[:2],  # Only x,y
            orientation=np.array([0, 0, 0]),  # No rotation initially
            side=1  # Right foot
        )
        
        self.footsteps = [left_footstep, right_footstep]
        self.current_support = Support(
            contacts=[left_footstep, right_footstep],
            foot_width=0.1,  # Approximate foot width
            foot_length=0.25  # Approximate foot length
        )
        
    def compute_capture_point(self, com, com_vel):
        """Compute capture point based on current CoM state, with height check"""
        min_height = 0.05
        h = max(com[2], min_height)
        w = np.sqrt(9.81 / h)
        cp = com[:2] + com_vel[:2] / w
        return cp
        
    def is_falling(self, com, com_vel):
        """Check if robot is falling by comparing capture point to support polygon"""
        if self.recovery_in_progress:
            return False  # Don't plan another step while recovering
        cp = self.compute_capture_point(com, com_vel)
        support_polygon = self.current_support.get_support_polygon()
        
        # Simple check: if CP is outside the convex hull of support points
        if len(support_polygon) >= 3:
            # Check if CP is inside the polygon (simplified)
            # For now, just check if CP is far from the center of support
            support_center = np.mean(support_polygon, axis=0)
            distance = np.linalg.norm(cp - support_center)
            
            # If distance is large, robot is falling
            if distance > 0.1:  # 10cm threshold
                return True
                
        return False
        
    def plan_recovery_step(self, com, com_vel):
        """Plan a recovery step to the capture point, clamped to max step length/width"""
        cp = self.compute_capture_point(com, com_vel)
        # Clamp step to max step length/width
        last_foot = self.footsteps[-1]
        delta = cp - last_foot.position
        max_length = 0.3  # meters
        max_width = 0.2   # meters
        delta[0] = np.clip(delta[0], -max_length, max_length)
        delta[1] = np.clip(delta[1], -max_width, max_width)
        clamped_cp = last_foot.position + delta
        # Determine which foot to step with (alternate)
        current_foot_side = self.footsteps[-1].side
        next_foot_side = 1 - current_foot_side  # Switch foot
        new_footstep = Footstep(
            position=clamped_cp,
            orientation=np.array([0, 0, 0]),
            side=next_foot_side
        )
        self.footsteps.append(new_footstep)
        return new_footstep
        
    def update_support_state(self, t):
        """Update the current support state based on stepping phase"""
        if self.step_phase == "double_support":
            # Check if we need to start a step
            if len(self.footsteps) > 2:  # We have planned steps
                self.step_phase = "single_support"
                self.step_start_time = t
                # Remove the foot that's about to step
                stepping_foot = self.footsteps[2]  # Next footstep
                if stepping_foot.side == 0:  # Left foot stepping
                    self.biped.removeLeftFootContact()
                else:  # Right foot stepping
                    self.biped.removeRightFootContact()
                    
        elif self.step_phase == "single_support":
            # Check if step is complete
            if t - self.step_start_time >= self.step_duration:
                self.step_phase = "double_support"
                self.recovery_in_progress = False  # Reset flag after step
                # Re-add the foot contact
                stepping_foot = self.footsteps[2]  # The foot that just stepped
                if stepping_foot.side == 0:  # Left foot
                    self.biped.addLeftFootContact()
                else:  # Right foot
                    self.biped.addRightFootContact()
                    
                # Update support
                self.current_support = Support(
                    contacts=[self.footsteps[1], self.footsteps[2]],  # Current and new foot
                    foot_width=0.1,
                    foot_length=0.25
                )
                
                # Remove the old footstep
                self.footsteps.pop(0)
                
    def get_foot_trajectory(self, t):
        """Get the current foot trajectory for stepping"""
        if self.step_phase == "single_support":
            stepping_foot = self.footsteps[2]  # Next footstep
            current_foot = self.footsteps[1]   # Current foot
            
            # Create foot trajectory
            start_pos = np.array([current_foot.position[0], current_foot.position[1], 0])
            target_pos = np.array([stepping_foot.position[0], stepping_foot.position[1], 0])
            
            foot_traj = FootTrajectory(
                t=[self.step_start_time, self.step_start_time + self.step_duration],
                start=start_pos,
                target=target_pos,
                step_height=step_height,
                rise_ratio=rise_ratio
            )
            
            return foot_traj
        return None

# Initialize the biped and controller
biped = Biped(conf)
controller = EnhancedBalanceController(biped, conf)

# Initialize the MuJoCo model and data
mj_model = mujoco.MjModel.from_xml_path(conf.mujoco_model_path)
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = conf.dt

push_robot_active = True
push_robot_com_vel = np.array([0.0, -0.2, 0.0])  # Stronger push

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

print("Enhanced Balance Controller Initialized!")
print("Using ctrl library for footstep planning and push recovery")

# Main simulation loop
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        t_elapsed = time.time() - start_time

        # Compute CoM reference and apply sinusoidal modification
        com_offset_x = amp * np.sin(2 * np.pi * freq * t)

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
            print("Simulation stopped due to QP failure.")
            break

        tau = biped.formulation.getActuatorForces(sol)
        dv = biped.formulation.getAccelerations(sol)
        q, v = biped.integrate_dv(q, v, dv, conf.dt)
        i, t = i + 1, t + conf.dt

        # Get center of mass state
        com = biped.robot.com(biped.formulation.data())
        com_vel = biped.robot.com_vel(biped.formulation.data())
        cp = controller.compute_capture_point(com, com_vel)

        # Check if robot is falling and plan recovery
        if controller.is_falling(com, com_vel):
            print(f"Robot falling detected at t={t:.2f}s! Planning recovery step...")
            recovery_footstep = controller.plan_recovery_step(com, com_vel)
            controller.recovery_in_progress = True
            print(f"Planned recovery step to: {recovery_footstep.position}")

        # Update support state and foot trajectories
        controller.update_support_state(t)
        
        # Update foot trajectories if stepping
        foot_traj = controller.get_foot_trajectory(t)
        if foot_traj is not None:
            current_time = t - controller.step_start_time
            if current_time <= controller.step_duration:
                foot_pos = foot_traj.get_position(current_time)
                # Update the appropriate foot trajectory
                stepping_foot = controller.footsteps[2]
                if stepping_foot.side == 0:  # Left foot
                    H_foot = pin.SE3(np.eye(3), np.array([foot_pos[0], foot_pos[1], foot_pos[2]]))
                    biped.trajLF.setReference(H_foot)
                else:  # Right foot
                    H_foot = pin.SE3(np.eye(3), np.array([foot_pos[0], foot_pos[1], foot_pos[2]]))
                    biped.trajRF.setReference(H_foot)

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
            pos=np.array([cp[0], cp[1], 0.0]),
            mat=np.eye(3).flatten(),
            rgba=np.array([1.0, 1.0, 0.0, 1.0]),
        )
        
        viewer.user_scn.ngeom = 5

        mj_data.ctrl = map_tsid_to_mujoco(q)
        mujoco.mj_step(mj_model, mj_data)

        # Apply push after 5 seconds
        if t_elapsed > 5.0 and push_robot_active:
            print("Applying push to robot!")
            push_robot_active = False
            data = biped.formulation.data()
            J_LF = biped.contactLF.computeMotionTask(0.0, q, v, data).matrix
            J_RF = biped.contactRF.computeMotionTask(0.0, q, v, data).matrix
            J = np.vstack((J_LF, J_RF))
            J_com = biped.comTask.compute(t, q, v, data).matrix
            A = np.vstack((J_com, J))
            b = np.concatenate((np.array(push_robot_com_vel), np.zeros(J.shape[0])))
            v = np.linalg.lstsq(A, b, rcond=-1)[0]

        viewer.sync()

    while viewer.is_running():
        viewer.sync()
        time.sleep(1.0)

print("Enhanced simulation finished!") 