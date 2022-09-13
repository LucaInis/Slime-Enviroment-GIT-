#from environments.SlimeEnvSingleAgent import Slime

import datetime
import gym
import json
import numpy as np
import random

PARAMS_FILE = "single-agent-params.json"
LEARNING_PARAMS_FILE = "sa-learning-params.json"
with open(LEARNING_PARAMS_FILE) as f:
    l_params = json.load(f)
OUTPUT_FILE = f"{l_params['OUTPUT_FILE']}-{datetime.datetime.now()}.csv"
with open(PARAMS_FILE) as f:
    params = json.load(f)
#env = Slime(render_mode="human", **params)
env = gym.make("unimore/SlimeEnvSingleAgent-v0", **params)

# Q-Learning
alpha = l_params["alpha"]  # DOC learning rate (0 learn nothing 1 learn suddenly)
gamma = l_params["gamma"]  # DOC discount factor (0 care only bout immediate rewards, 1 care only about future ones)
epsilon = l_params["epsilon"]  # DOC chance of random action
decay = l_params["decay"]  # DOC di quanto diminuisce epsilon ogni episode (e.g. 1500 episodes => decay = 0.9995)
TRAIN_EPISODES = l_params["train_episodes"]
TEST_EPISODES = l_params["test_episodes"]
TRAIN_LOG_EVERY = l_params["TRAIN_LOG_EVERY"]
TEST_LOG_EVERY = l_params["TEST_LOG_EVERY"]

with open(OUTPUT_FILE, 'w') as f:
    f.write(f"{json.dumps(params, indent=2)}\n")
    f.write("----------\n")
    f.write(f"TRAIN_EPISODES = {TRAIN_EPISODES}\n")
    f.write(f"TEST_EPISODES = {TEST_EPISODES}\n")
    f.write("----------\n")
    f.write(f"alpha = {alpha}\n")
    f.write(f"gamma = {gamma}\n")
    f.write(f"epsilon = {epsilon}\n")
    f.write(f"decay = {decay}\n")
    f.write("----------\n")
    # From NetlogoDataAnalysis: Episode, Tick, Avg cluster size X tick, Avg reward X episode, move-toward-chemical, random-walk, drop-chemical, (learner 0)-move-toward-chemical
    f.write(f"Episode, Tick, Avg cluster size X tick, ")
    for a in l_params["actions"]:
        f.write(f"{a}, ")
    f.write("Avg reward X episode\n")

q_table = np.zeros([4, env.action_space.n])

# DOC dict che tiene conto della frequenza di scelta delle action per ogni episodio {episode: {action: _, action: _, ...}}
actions_dict = {str(ep): {str(ac): 0 for ac in range(3)} for ep in range(1, TRAIN_EPISODES+1)}  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
# DOC dict che tiene conto della reward per ogni episodio {episode: _}
reward_dict = {str(ep): 0 for ep in range(1, TRAIN_EPISODES+1)}
# DOC dict che tiene conto della dimensioni di ogni cluster per ogni episodio
cluster_dict = {}


def state_to_int_map(obs: [bool, bool]):
    if sum(obs) == 0:  # DOC [False, False]
        mapped = sum(obs)  # 0
    elif sum(obs) == 2:  # DOC [True, True]
        mapped = 3
    elif int(obs[0]) == 1 and int(obs[1]) == 0:  # DOC [True, False] ==> si trova in un cluster ma non su una patch con feromone --> difficile succeda
        mapped = 1
    else:
        mapped = 2  # DOC [False, True]
    return mapped


# TRAINING
print("Start training...")
for ep in range(1, TRAIN_EPISODES+1):
    state, _ = env.reset()
    s = state_to_int_map(state)
    for tick in range(1, params['episode_ticks']+1):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[s])  # Exploit learned values

        next_state, reward, _, _, _ = env.step(action)
        next_s = state_to_int_map(next_state)

        old_value = q_table[s][action]
        next_max = np.max(q_table[next_s])  # QUESTION: was with [s]

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[s][action] = new_value

        s = next_s

        actions_dict[str(ep)][str(action)] += 1
        reward_dict[str(ep)] += round(reward, 2)

        env.render()
    epsilon *= decay
    #cluster_dict[str(ep)] = round(env.avg_cluster(), 2)
    if ep % TRAIN_LOG_EVERY == 0:
        print(f"EPISODE: {ep}")
        print(f"\tepsilon: {epsilon}")
        print(f"\tq_table: {q_table}")
        with open(OUTPUT_FILE, 'a') as f:
            f.write(
                f"{ep}, {params['episode_ticks'] * ep}, {cluster_dict[str(ep)]}, {actions_dict[str(ep)]['2']}, {actions_dict[str(ep)]['0']}, {actions_dict[str(ep)]['1']}, ")
            f.write(f"{reward_dict[str(ep)]}\n")

#print(json.dumps(cluster_dict, indent=2))
print("Training finished!\n")

# DOC Evaluate agent's performance after Q-learning
cluster_dict = {}
print("Start testing...")
for ep in range(1, TEST_EPISODES+1):
    reward_episode = 0
    state, _ = env.reset()
    state = sum(state)
    for tick in range(params['episode_ticks']):
        action = np.argmax(q_table[state])
        state, reward, _, _, _ = env.step(action)
        state = sum(state)
        reward_episode += reward
        env.render()
    if ep % TEST_LOG_EVERY == 0:
        print(f"EPISODE: {ep}")
        print(f"\tepisode reward: {reward_episode}")
    #cluster_dict[str(ep)] = round(env.avg_cluster(), 2)
print(json.dumps(cluster_dict, indent=2))
print("Testing finished!\n")
env.close()
