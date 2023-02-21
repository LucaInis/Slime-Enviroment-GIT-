from stable_baselines3.common.env_checker import check_env
import slime_environments
import gym
import json

PARAMS_FILE = "slime_environments/agents/single-agent-params.json"
with open(PARAMS_FILE) as f:
    params = json.load(f)
env = gym.make("Slime-v0", **params)
# env = Slime(**params)
# check_env(env)
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
obs = MultiBinary(2)
print(obs.sample())