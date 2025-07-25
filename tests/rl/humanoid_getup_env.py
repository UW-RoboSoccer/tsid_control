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

        self.previous_action = np.zeros(self.model.nu)

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def _get_reward(self):
        TORSO_BODY_ID = 1
        HEAD_BODY_ID = 21
        HEAD_TARGET_HEIGHT = 0.45

        RIGHT_FOOT_ID = 7
        LEFT_FOOT_ID = 13
        WORLD_ID = 0

        h_head = self.data.xpos[HEAD_BODY_ID][2]
        xmat = self.data.xmat[TORSO_BODY_ID].reshape(3, 3)
        pitch = np.arcsin(-xmat[2, 0])

        # Strong head height reward
        R_up = 3.0 * np.clip(h_head / HEAD_TARGET_HEIGHT, 0.0, 1.0)

        # Pitch alignment reward (only when head is high enough)
        R_pitch = 1.0 * float(h_head > 0.3) * np.exp(-10.0 * pitch ** 2)

        # Control effort reward (only once upright-ish)
        delta_a = self.data.ctrl - self.previous_action
        R_var = 0.05 * np.exp(-np.linalg.norm(delta_a)) if h_head > 0.35 else 0.0
        self.previous_action = self.data.ctrl.copy()

        # Velocity reward (only if near standing)
        R_vel = 0.05 * np.exp(-np.linalg.norm(self.data.qvel)) if h_head > 0.35 else 0.0

        # Contact info
        self_collisions = 0
        contact_right = 0
        contact_left = 0

        try:
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                b1 = self.model.geom_bodyid[c.geom1]
                b2 = self.model.geom_bodyid[c.geom2]

                if b1 == b2 and b1 != 0:
                    self_collisions += 1

                if WORLD_ID in [b1, b2]:
                    if RIGHT_FOOT_ID in [b1, b2]:
                        contact_right = 1
                    if LEFT_FOOT_ID in [b1, b2]:
                        contact_left = 1
        except Exception as e:
            if self.debug:
                print(f"[REWARD DEBUG] Contact processing error: {e}")

        R_coll = 0.1 * np.exp(-self_collisions)
        R_foot_contact = 0.1 * (contact_right + contact_left)

        # Bonus for staying upright
        bonus = 0.0
        if h_head > 0.43 and abs(pitch) < 0.3:
            bonus += 3.0
            if self._standing_counter % int(1.0 / self.dt) == 0:
                bonus += 3.0

        # Mild penalty for being too low
        penalty = -0.5 if h_head < 0.15 else 0.0

        reward = (
            R_up + R_pitch + R_vel + R_var + R_coll +
            R_foot_contact + bonus + penalty
        )

        if self.debug:
            print(f"[REWARD DEBUG] Head Height: {h_head:.3f}, Pitch: {pitch:.3f}")
            print(f"[REWARD DEBUG] R_up: {R_up:.3f}, R_pitch: {R_pitch:.3f}, R_vel: {R_vel:.3f}, "
                f"R_var: {R_var:.3f}, R_coll: {R_coll:.3f}, R_foot_contact: {R_foot_contact:.3f}")
            print(f"[REWARD DEBUG] Bonus: {bonus:.3f}, Penalty: {penalty:.3f}, Total: {reward:.3f}")

        return reward






    def _is_done(self):
        self._step_counter += 1
        h_head = self.data.xpos[21][2]
        uprightness = self.data.xmat[1].reshape(3, 3)[:, 2] @ np.array([0, 0, 1])
        low_vel = np.linalg.norm(self.data.qvel[:3]) < 0.5

        if h_head > 0.43 and uprightness > 0.9 and low_vel:
            self._standing_counter += 1
        else:
            self._standing_counter = 0

        if self._standing_counter >= self._standing_threshold:
            return True
        if self._step_counter >= self._max_steps:
            return True
        if np.any(np.abs(self.data.qvel) > 25.0):
            return True
        return False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.data.time = 0.0

        self._step_counter = 0
        self._standing_counter = 0
        self.previous_action = np.zeros(self.model.nu)

        self.data.qpos[0:3] = np.random.uniform(-0.1, 0.1, size=3)
        angle = np.random.uniform(-np.pi / 2, np.pi / 2)
        quat = [np.cos(angle / 2), np.sin(angle / 2), 0, 0]
        self.data.qpos[3:7] = quat
        self.data.qpos[7:] = np.random.uniform(-1.0, 1.0, size=self.model.nq - 7)
        self.data.qvel[:] = np.random.normal(0, 0.1, size=self.model.nv)

        mujoco.mj_forward(self.model, self.data)
        if self.render_mode and self.viewer is not None:
            self.viewer.sync()
        print("[ENV RESET] Resetting environment with new random pose.")
        return self._get_obs(), {}

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
            "is_success": bool(self._standing_counter >= self._standing_threshold)
        }

        return obs, reward, done, False, info




    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
