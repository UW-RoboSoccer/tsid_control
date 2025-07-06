import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer

class HumanoidGetupEnv(gym.Env):
    def __init__(self, render_mode=False):
        super().__init__()
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path("../../robot/v1/mujoco/robot_damped.xml")
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        self.viewer = None
        if self.render_mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Observation = joint positions, velocities, torso height/orientation
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Action = joint torques (clip to actuator limits)
        act_dim = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

        self.sim_steps_per_action = 10  # More stable training
        self._step_counter = 0
        self._standing_counter = 0
        self._standing_threshold = int(10 / self.dt)  # 10 seconds
        self._max_steps = int(10 / self.dt)  # 10 seconds

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def _get_reward(self):
        reward = 0.0

        # --------------------------------------
        # Get torso position, orientation, and velocity
        # --------------------------------------
        torso_height = self.data.qpos[2]                # z-pos of torso root
        torso_quat = self.data.qpos[3:7]                # orientation quaternion
        torso_upright_score = torso_quat[0]             # 'w' component ~1 when upright
        torso_z_vel = self.data.qvel[2]                 # vertical velocity

        # --------------------------------------------------
        # Reward standing upright (if above minimum height)
        # --------------------------------------------------
        if torso_height > 0.6:                          # adjust threshold as needed
            reward += 5.0 * torso_upright_score         # reward being upright
            reward += 2.0                               # bonus for height itself
        else:
            reward -= 1.0                               # penalty if still on ground

        # Penalize vertical bouncing to reduce flailing
        reward -= 1.0 * abs(torso_z_vel)

        # --------------------------------------------------
        # Contact-based reward/penalty (feet vs torso)
        # --------------------------------------------------
        ground_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        torso_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "torso")
        foot_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "foot2 ")
        ]

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1, g2 = contact.geom1, contact.geom2

            # Reward feet-ground contact
            if ground_id in (g1, g2) and (g1 in foot_ids or g2 in foot_ids):
                reward += 1.0

            # Penalize torso-ground contact
            if (g1 == torso_id and g2 == ground_id) or (g2 == torso_id and g1 == ground_id):
                reward -= 3.0

        return reward





    def _is_standing(self):
        torso_height = self.data.qpos[2]
        torso_quat = self.data.qpos[3:7]
        torso_up = torso_quat[0]  # w close to 1 when upright
        low_vel = np.linalg.norm(self.data.qvel[:3]) < 0.5

        return torso_height > 0.9 and torso_up > 0.9 and low_vel

    def _is_done(self):
        # How long we've been running
        self._step_counter += 1
        # Measure torso height and orientation
        torso_height = self.data.qpos[2]
        up_vector = self.data.xmat[1].reshape(3, 3)[:, 2]
        uprightness = up_vector @ np.array([0, 0, 1])  # dot product with global Z
        # Check if standing
        is_upright = torso_height > 0.8 and uprightness > 0.9
        if is_upright:
            self._standing_counter += 1
        else:
            self._standing_counter = 0
        # End if success: standing for 10 seconds
        if self._standing_counter >= self._standing_threshold:
            return True
        # End if episode is too long (fail timeout)
        if self._step_counter >= self._max_steps:
            return True
        # Do NOT end if it's just fallen allow recovery!
        return False


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Reset simulation state
        mujoco.mj_resetData(self.model, self.data)
        # Reset counters
        self._step_counter = 0
        self._standing_counter = 0
        # Position: torso slightly above ground
        self.data.qpos[0:3] = [0, 0, 0.06]  # x, y, z
        # Orientation: 90° rotation around X-axis (lying flat on back)
        self.data.qpos[3:7] = [0.7071, 0.7071, 0.0, 0.0]  # w, x, y, z
        # Joint positions: collapsed pose
        self.data.qpos[7:19] = np.deg2rad([
            30, -20, -45, 90, -10, 5,   # right leg
        -30,  20,  45, 90,  10, -5   # left leg
        ])
        # Zero velocities
        self.data.qvel[:] = 0
        # Recompute derived quantities
        mujoco.mj_forward(self.model, self.data)
        # Get initial observation
        obs = self._get_obs()
        return obs, {}
    
        # super().reset(seed=seed)  # Ensures reproducibility
        # mujoco.mj_resetData(self.model, self.data)
        # self.data.qpos[:] = np.zeros_like(self.data.qpos)
        # self.data.qvel[:] = np.zeros_like(self.data.qvel)
        # self.data.qpos[2] = 0.3  # Low height
        # obs = self._get_obs()
        # info = {}
        # return obs, info

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        ctrl_range = self.model.actuator_ctrlrange
        ctrl_low = ctrl_range[:, 0]
        ctrl_high = ctrl_range[:, 1]
        scaled_action = ctrl_low + (action + 1.0) * 0.5 * (ctrl_high - ctrl_low)
        self.data.ctrl[:] = scaled_action

        for _ in range(self.sim_steps_per_action):
            mujoco.mj_step(self.model, self.data)

        if self.render_mode and self.viewer is not None:
            self.viewer.sync()

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_done()   # End due to task failure (falling)
        truncated = False              # Can implement time limit later
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
