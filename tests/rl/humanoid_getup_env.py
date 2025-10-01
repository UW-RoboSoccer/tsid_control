# fixed_humanoid_getup_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer

class HumanoidGetupEnv(gym.Env):
    def __init__(self, render_mode=False, debug=False):
        super().__init__()
        self.render_mode = render_mode
        self.debug = debug

        self.model = mujoco.MjModel.from_xml_path("../../robot/v1/mujoco/robot_damped.xml")
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        self.viewer = None
        if self.render_mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.selected_indices = [9, 10, 11, 16, 17, 19, 20, 21]  # shoulder_pitch, elbow, hip_pitch, knee, ankle_pitch

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.model.nq + self.model.nv,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(5,),  # 5 symmetric DOFs
            dtype=np.float32
        )


        self.sim_steps_per_action = 10
        self._step_counter = 0
        self._standing_counter = 0
        self._standing_threshold = int(1.0 / self.dt)
        self._max_steps = int(10.0 / self.dt)
        self.best_height = 0.0        # for progress shaping
        self.best_upright = 0.0
        self.no_progress_steps = 0    # to detect stalls


        self.previous_action = np.zeros(self.model.nu)

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def _seg_upright(self, proximal_body_name: str, distal_body_name: str) -> float:
        bp = self.model.body(proximal_body_name).id
        bd = self.model.body(distal_body_name).id
        v = self.data.xpos[bd] - self.data.xpos[bp]     # world vector along the segment
        n = v / (np.linalg.norm(v) + 1e-9)
        if float(self.data.xpos[self.model.body(proximal_body_name).id][2]) > float(self.data.xpos[self.model.body(distal_body_name).id][2]):
            return float(n[2] * n[2])  # 0..1 (1 = perfectly vertical)
        else:
            return -float(n[2] * n[2])  # 0..1 (1 = perfectly vertical)
    
    def foot_flatness(self, body_name):
        R = self.data.xmat[self.model.body(body_name).id].reshape(3,3)
        ez = R[:,2]                        # body +Z (choose the axis that points out of the sole)
        return abs(ez @ np.array([0,1,0])) # 1 when normal aligned with world-Z

    def com_over_feet_score(self) -> float:
        LEFT_FOOT_ID, RIGHT_FOOT_ID = 13, 7
        com = np.array(self.data.subtree_com[1][:2])

        lf = self.data.xpos[LEFT_FOOT_ID][:2]
        rf = self.data.xpos[RIGHT_FOOT_ID][:2]

        # Project COM to the line segment between feet
        v  = rf - lf
        vv = np.dot(v, v) + 1e-9
        t  = np.clip(np.dot(com - lf, v) / vv, 0.0, 1.0)
        p  = lf + t * v                       # closest point on segment
        dist_lat = np.linalg.norm(com - p)    # sideways distance to segment

        mid = 0.5 * (lf + rf)
        width = np.sqrt(vv)
        # also keep COM near the midpoint along the segment
        off_along = abs(np.dot(com - mid, v) / (width + 1e-9))

        sigma_lat   = 0.15 * width + 0.03
        sigma_along = 0.20 * width + 0.04
        return float(np.exp(-(dist_lat / sigma_lat)**2) * np.exp(-(off_along / sigma_along)**2))

    def head_vertical_over_feet(self) -> float:
        LF, RF, HEAD = 13, 7, 21
        mid = 0.5 * (self.data.xpos[LF] + self.data.xpos[RF])   # midpoint of feet (world)
        v   = self.data.xpos[HEAD] - mid                        # feet->head (world)
        n   = v / (np.linalg.norm(v) + 1e-9)
        return float(max(0.0, n[2])**2)  # only reward “up”, not down


    # def _get_reward(self):
    #     # ids
    #     TORSO_BODY_ID = 1
    #     HEAD_BODY_ID = 21
    #     RIGHT_FOOT_ID = 7
    #     LEFT_FOOT_ID = 13
    #     RIGHT_HAND_BODY = 16
    #     LEFT_HAND_BODY  = 19
    #     WORLD_ID = 0
    #     BUTT_ID = 18

    #     # constants
    #     EARLY_USE_HANDS_UNTIL = 0.32

    #     h_head = float(self.data.xpos[HEAD_BODY_ID][2])
    #     h_foot = float(self.data.xpos[LEFT_FOOT_ID][2])
    #     xmat = self.data.xmat[TORSO_BODY_ID].reshape(3, 3)
    #     uprightness = float(xmat[:, 2] @ np.array([0.0, 0.0, 1.0]))  # 0..1

    #     R_foot = 0.0
    #     R_head_vertical = 0.0
        
    #     # tracking contact with floor
    #     feet_contact = 0
    #     hand_contact = 0
    #     unsafe_contacts = 0
    #     for i in range(self.data.ncon):
    #         c = self.data.contact[i]
    #         b1 = self.model.geom_bodyid[c.geom1]
    #         b2 = self.model.geom_bodyid[c.geom2]
    #         # print("b1: " + str(b1))
    #         # print("b2: " + str(b2))
    #         if WORLD_ID in (b1, b2):
    #             if RIGHT_FOOT_ID in (b1, b2) or LEFT_FOOT_ID in (b1, b2):
    #                 feet_contact = 1
    #                 if h_foot < 0.03:
    #                     R_foot = 20.0 * self.foot_flatness("foot")
    #                     R_head_vertical = 20.0 * self.head_vertical_over_feet()
    #                 R_foot += 20.0 * self.com_over_feet_score()
    #             if RIGHT_HAND_BODY in (b1, b2) or LEFT_HAND_BODY in (b1, b2):
    #                 hand_contact = 1
    #             # mild penalty for any other world contact (torso, head, upper arm, etc.)
    #             if (b1 not in (RIGHT_FOOT_ID, LEFT_FOOT_ID, RIGHT_HAND_BODY, LEFT_HAND_BODY, WORLD_ID) or
    #                 b2 not in (RIGHT_FOOT_ID, LEFT_FOOT_ID, RIGHT_HAND_BODY, LEFT_HAND_BODY, WORLD_ID)):
    #                 unsafe_contacts += 1
    #                 if(b2 == HEAD_BODY_ID or b2 == TORSO_BODY_ID):
    #                     unsafe_contacts += 2
    #     # print(self.head_vertical_over_feet())
    #     # tracking leg uprightness
    #     thigh_R = self._seg_upright("knee_housing", "knee_ankle_rotor_case")
    #     shank_R = self._seg_upright("knee_ankle_rotor_case", "sts3250")
    #     thigh_L = self._seg_upright("sts3250_2", "knee_ankle_rotor_case_2")
    #     shank_L = self._seg_upright("knee_ankle_rotor_case_2", "sts3250_3")

    #     # reward for legs being straight and up
    #     R_leg = 10.0 * (thigh_R + shank_R + thigh_L + shank_L) 

    #     # reward for higher head height
    #     if h_head < h_foot:
    #         R_height = -500
    #     else:
    #         R_height = 100.0 * h_head

    #     # butt height
    #     # R_butt = 300.0 * float(self.data.xpos[self.model.body("sts3250_4").id][2]) + 300.0 * float(self.data.xpos[self.model.body("sts3250_2").id][2])
    #     # R_butt = 0.0

    #     R_upright = 0.0
    #     if h_head > 0.2:
    #         R_upright = 100.0 * uprightness

    #     # reward for feet touching and hands touching, punish for hands on floor after head height is above threshold (push off floor)
    #     R_contacts = 0.0
    #     if h_head < EARLY_USE_HANDS_UNTIL:
    #         R_contacts += 10.0 * hand_contact
    #         R_contacts += 20.0 * feet_contact
    #     else:
    #         R_contacts += 50.0 * feet_contact
    #         R_contacts += -20.0 * hand_contact

    #     # punishes any contact on the floor except hands and feet
    #     R_unsafe = -30.0 * unsafe_contacts

    #     # print(self.com_over_feet_score())
    #     # print(leg_upright)
    #     # print(self.foot_flatness("foot"))
    #     # print("sts3250: " + str(float(self.data.xpos[self.model.body("sts3250").id][2])))
    #     # print("sts3250_2: " + str(float(self.data.xpos[self.model.body("sts3250_2").id][2])))
    #     # print("sts3250_3: " + str(float(self.data.xpos[self.model.body("sts3250_3").id][2])))
    #     # print("sts3250_4: " + str(float(self.data.xpos[self.model.body("sts3250_4").id][2])))
    #     # print("sts3250_5: " + str(float(self.data.xpos[self.model.body("sts3250_5").id][2])))
    #     # print("sts3250_6: " + str(float(self.data.xpos[self.model.body("sts3250_6").id][2])))
    #     # print("sts3250_4: " + str(self.model.body("sts3250_4").id))

    #     reward = R_height + R_upright + R_contacts + R_unsafe + R_leg + R_foot + R_head_vertical

    #     return reward

    def _get_reward(self):
        # ids
        TORSO_BODY_ID = 1
        HEAD_BODY_ID = 21
        RIGHT_FOOT_ID = 7
        LEFT_FOOT_ID = 13
        RIGHT_HAND_BODY = 16
        LEFT_HAND_BODY  = 19
        WORLD_ID = 0
        BUTT_ID = 18

        # constants
        EARLY_USE_HANDS_UNTIL = 0.32
        HEIGHT = 0.45

        h_head = float(self.data.xpos[HEAD_BODY_ID][2])
        h_foot = float(self.data.xpos[LEFT_FOOT_ID][2])
        xmat = self.data.xmat[TORSO_BODY_ID].reshape(3, 3)
        uprightness = float(xmat[:, 2] @ np.array([0.0, 0.0, 1.0]))  # 0..1

        thigh_R = self._seg_upright("knee_housing", "knee_ankle_rotor_case")
        shank_R = self._seg_upright("knee_ankle_rotor_case", "sts3250")
        thigh_L = self._seg_upright("sts3250_2", "knee_ankle_rotor_case_2")
        shank_L = self._seg_upright("knee_ankle_rotor_case_2", "sts3250_3")

        R_foot = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            # print("b1: " + str(b1))
            # print("b2: " + str(b2))
            if WORLD_ID in (b1, b2):
                if (RIGHT_FOOT_ID in (b1, b2) or LEFT_FOOT_ID in (b1, b2)) and h_head > h_foot:
                    R_foot = np.exp(-10 * pow(self.foot_flatness("foot"), 2)) + np.exp(-10 * pow(thigh_R + shank_R + thigh_L + shank_L, 2)) 
        

        R_up = np.exp(-10 * pow(abs(h_head - HEIGHT), 2))
        R_pitch = np.exp(-10 * pow(uprightness, 2)) if h_head > 0.4 * HEIGHT else 0.0
        # R_collision = 0.01 * np.exp(-self_collisions)
        R_foot = np.exp(-10 * pow(abs(1 - self.foot_flatness("foot")), 2)) + 10 * (h_head - h_foot)

        reward = R_up + R_pitch + R_foot
        return reward

    def _is_done(self):
        self._step_counter += 1
        h_head = self.data.xpos[21][2]
        uprightness = self.data.xmat[1].reshape(3, 3)[:, 2] @ np.array([0, 0, 1])
        low_vel = np.linalg.norm(self.data.qvel[:3]) < 0.5

        if h_head > 0.43 and uprightness > 0.9:
            self._standing_counter += 1
        else:
            self._standing_counter = 0

        if self._standing_counter >= self._standing_threshold:
            return True
        if self._step_counter >= self._max_steps:
            return True
        # if np.any(np.abs(self.data.qvel) > 25.0):
        #     return True
        return False

    # back down
    # def reset(self, *, seed=None, options=None):
    #     super().reset(seed=seed)

    #     mujoco.mj_resetData(self.model, self.data)
    #     self.data.time = 0.0

    #     self._step_counter = 0
    #     self._standing_counter = 0
    #     self.previous_action = np.zeros(self.model.nu)

    #     # ---- Pose: belly-down ----
    #     self.data.qpos[0:2] = [0.0, 0.0]
    #     self.data.qpos[2] = 0.08  # low height

    #     # Pitch forward 90° to lie on stomach
    #     angle = -np.pi / 2
    #     quat = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]  # pitch only
    #     self.data.qpos[3:7] = quat

    #     # Neutral joints
    #     self.data.qpos[7:] = np.zeros(self.model.nq - 7)
    #     self.data.qvel[:] = np.zeros(self.model.nv)

    #     mujoco.mj_forward(self.model, self.data)

    #     if self.render_mode and self.viewer is not None:
    #         self.viewer.sync()

    #     print("[ENV RESET] Humanoid reset to back-down position.")
    #     return self._get_obs(), {}

    # belly down
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.time = 0.0

        self._step_counter = 0
        self._standing_counter = 0
        self.previous_action = np.zeros(self.model.nu)
        mujoco.mj_forward(self.model, self.data)
        self.best_height = float(self.data.xpos[21][2])  # head z at reset
        self.no_progress_steps = 0


        # Set base position
        self.data.qpos[0:2] = [0.0, 0.0]       # x, y position
        self.data.qpos[2] = 0.0              # z height close to ground

        # Pitch forward 90° = face/belly down
        angle = np.pi / 2
        quat = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]
        self.data.qpos[3:7] = quat

        # Neutral limbs
        self.data.qpos[7:] = np.zeros(self.model.nq - 7)
        self.data.qvel[:] = np.zeros(self.model.nv)

        mujoco.mj_forward(self.model, self.data)

        if self.render_mode and self.viewer is not None:
            self.viewer.sync()

        print("[ENV RESET] Humanoid reset belly-down (face-down).")
        return self._get_obs(), {}



    # def reset(self, *, seed=None, options=None):
    #     super().reset(seed=seed)

    #     mujoco.mj_resetData(self.model, self.data)
    #     self.data.time = 0.0

    #     self._step_counter = 0
    #     self._standing_counter = 0
    #     self.previous_action = np.zeros(self.model.nu)

    #     # ---- Pose: upright standing ----
    #     self.data.qpos[0:2] = [0.0, 0.0]         # x, y base position
    #     self.data.qpos[2] = .250                  # z: height above ground

    #     self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity quaternion = upright

    #     self.data.qpos[7:] = np.zeros(self.model.nq - 7)           # Neutral joints
    #     self.data.qvel[:] = np.zeros(self.model.nv)                # No velocity

    #     mujoco.mj_forward(self.model, self.data)

    #     if self.render_mode and self.viewer is not None:
    #         self.viewer.sync()

    #     print("[ENV RESET] Humanoid reset upright (standing).")
    #     return self._get_obs(), {}


    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        
        # (left_idx, right_idx, sign_flip_for_right)
        symmetric_actuators = [
            (8, 2, True),   # Hip pitch: no flip
            (9, 3, True),   # Knee: no flip
            (10, 4, True),   # Ankle pitch: flip
            (12, 15, True), # Shoulder pitch: no flip
            (14, 17, True), # Elbow: no flip
        ]
        
        full_action = np.zeros(self.model.nu)
        
        for i, (left_idx, right_idx, flip_right) in enumerate(symmetric_actuators):
            full_action[left_idx] = action[i]
            full_action[right_idx] = -action[i] if flip_right else action[i]

        # Convert to control signal
        ctrl_range = self.model.actuator_ctrlrange
        scaled = ctrl_range[:, 0] + (full_action + 1.0) * 0.5 * (ctrl_range[:, 1] - ctrl_range[:, 0])
        self.data.ctrl[:] = scaled

        for _ in range(self.sim_steps_per_action):
            mujoco.mj_step(self.model, self.data)

        if self.render_mode and self.viewer is not None:
            self.viewer.sync()

        obs = self._get_obs()
        reward = self._get_reward()
        done = self._is_done()
        info = {
            "is_success": bool(self._standing_counter >= self._standing_threshold),
            "standing_counter": self._standing_counter,
            "head_height": self.data.xpos[21][2],
            "pitch": np.arcsin(-self.data.xmat[1].reshape(3, 3)[:, 2] @ np.array([0, 0, 1]))
        }

        return obs, reward, done, False, info




    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()