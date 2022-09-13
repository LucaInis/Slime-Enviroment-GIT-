import json
import gym
import slime_environments
from gym.utils.env_checker import check_env

PARAMS_FILE = "single-agent-params.json"
with open(PARAMS_FILE) as f:
    params = json.load(f)
#env = Slime(render_mode="human", **params)
env = gym.make("Slime-v0", **params)

check_env(env.unwrapped, skip_render_check=False)
