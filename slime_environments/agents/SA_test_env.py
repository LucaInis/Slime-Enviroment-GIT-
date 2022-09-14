import json
from itertools import permutations, combinations, product

import gym
import slime_environments
from gym.utils.env_checker import check_env, check_reset_seed, check_reset_return_type

from slime_environments.environments.SlimeEnvSingleAgent import BooleanSpace

PARAMS_FILE = "single-agent-params.json"
with open(PARAMS_FILE) as f:
    params = json.load(f)
#env = Slime(render_mode="human", **params)
env = gym.make("Slime-v0", **params)

check_env(env.unwrapped, skip_render_check=False)
check_reset_seed(env)
check_reset_return_type(env)

# space = BooleanSpace(size=2)
# print(f"size={space.size}, shape={space.shape}, values={space._values}, sample={space.sample()}")
#
# print(list(permutations([True, False])))
# print(list(product([True, False], repeat=2)))
# print(list(combinations([x for x in [True, False]], 2)))
