from stable_baselines3.common.env_checker import check_env
import slime_environments
import gym
import json
from gym.spaces import MultiBinary

PARAMS_FILE = "slime_environments/agents/single-agent-params.json"
with open(PARAMS_FILE) as f:
    params = json.load(f)

# print(gym.__version__)
env = gym.make("Slime-v0", **params)
check_env(env)
print("Environment compatible with Stable Baselines3")
