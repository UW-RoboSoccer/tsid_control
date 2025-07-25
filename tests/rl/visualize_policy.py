from stable_baselines3 import PPO
from humanoid_getup_env import HumanoidGetupEnv
import time

# Load environment (render_mode=True to see MuJoCo viewer)
env = HumanoidGetupEnv(render_mode=True)

# Load trained model (or comment this out to test random policy)
model = PPO.load("models/ppo_humanoid_2000000_steps")

obs, _ = env.reset()

for step in range(5000):  # ~10 seconds of sim
    # Predict action
    action, _states = model.predict(obs, deterministic=True)
    # action = env.action_space.sample()  # random action

    # Take step in environment
    obs, reward, terminated, truncated, info = env.step(action)

    # Optional: slow down to real time
    time.sleep(env.dt)

    # Restart if episode ends
    if terminated or truncated:
        print("Episode ended. Resetting.")
        obs, _ = env.reset()
