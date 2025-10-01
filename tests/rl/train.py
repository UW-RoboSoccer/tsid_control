# from stable_baselines3 import PPO
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.callbacks import CheckpointCallback
# from humanoid_getup_env import HumanoidGetupEnv
# import os

# os.environ["MUJOCO_GL"] = "osmesa"

# env = HumanoidGetupEnv(render_mode=True)

# # Load previously trained model
# # model = PPO.load("./models/humanoid_getup_policy", env=env)

# obs, info = env.reset()
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs/")

# checkpoint_callback = CheckpointCallback(
#     save_freq=100_000,  # every 10k timesteps
#     save_path="./checkpoints/",
#     name_prefix="ppo_humanoid"
# )

# model.learn(total_timesteps=10_000_000, callback=checkpoint_callback)

# model.save("./models/humanoid_getup_policy")



from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from humanoid_getup_env import HumanoidGetupEnv
import torch

# (Optional) keep PyTorch from hogging CPU threads used by env workers
torch.set_num_threads(1)

def make_env(rank: int):
    def _init():
        # No rendering in workers; render only in eval scripts
        return HumanoidGetupEnv(render_mode=False)
    return _init

def main():
    n_envs = 8
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        device="cuda",   # or "cpu"
        verbose=1,
        tensorboard_log="./ppo_logs/",
        n_steps=2048,
        batch_size=n_envs * 64,  # must divide n_envs * n_steps
        n_epochs=10,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=102_400,  # multiple of n_envs
        save_path="./checkpoints/",
        name_prefix="ppo_humanoid",
    )

    model.learn(total_timesteps=10_000_000, callback=checkpoint_callback)
    model.save("./models/humanoid_getup_policy")
    env.close()

if __name__ == "__main__":
    main()
