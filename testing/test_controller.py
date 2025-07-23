import mujoco


class TestController:
    def __init__(self):
        pass

    def update_control(self, mj_model, mj_data):
        mj_data.ctrl[:] = 0.0

    def set_initial_pose(self, mj_model, mj_data):
        # Base pose
        mj_data.qpos[0:3] = [0, 0, 1.0]  # x, y, z base height (adjust as needed)
        mj_data.qpos[3:7] = [1, 0, 0, 0]  # identity quaternion

        # Right leg
        mj_data.qpos[7] = 0.0  # right_hip_yaw
        mj_data.qpos[8] = 0.0  # right_hip_roll
        mj_data.qpos[9] = 0.2  # right_hip_pitch
        mj_data.qpos[10] = -0.4  # right_knee
        mj_data.qpos[11] = 0.2  # right_ankle_pitch
        mj_data.qpos[12] = 0.0  # right_ankle_roll

        # Left leg
        mj_data.qpos[13] = 0.0  # left_hip_yaw
        mj_data.qpos[14] = 0.0  # left_hip_roll
        mj_data.qpos[15] = 0.2  # left_hip_pitch
        mj_data.qpos[16] = -0.4  # left_knee
        mj_data.qpos[17] = 0.2  # left_ankle_pitch
        mj_data.qpos[18] = 0.0  # left_ankle_roll

    def align_on_ground(self, mj_model, mj_data):
        foot_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "foot")
        foot_z = mj_data.xpos[foot_id][2]
        print(f"footz {foot_z}")
        mj_data.qpos[2] -= foot_z - 0.05
        print(mj_data.qpos[2])


test_controller = TestController()
