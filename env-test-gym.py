from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
import slime_environments
import gym
import json
from gym.spaces import MultiBinary
import numpy as np

PARAMS_FILE = "slime_environments/agents/single-agent-params.json"
with open(PARAMS_FILE) as f:
    params = json.load(f)

# print(gym.__version__)
env = gym.make("Slime-v0", **params)
check_env(env)
print("Environment compatible with Stable Baselines3")

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000,log_interval=4)
print("SB3 DQN sample training completed.")
