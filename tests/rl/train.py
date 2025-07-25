from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from humanoid_getup_env import HumanoidGetupEnv


env = HumanoidGetupEnv(render_mode=True, debug=True)

# Load previously trained model
# model = PPO.load("./models/humanoid_getup_policy", env=env)

obs, info = env.reset()
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs/")

checkpoint_callback = CheckpointCallback(
    save_freq=1_000_000,  # every 10k timesteps
    save_path="./checkpoints/",
    name_prefix="ppo_humanoid"
)

model.learn(total_timesteps=4_000_000, callback=checkpoint_callback)

model.save("./models/humanoid_getup_policy")