from SlimeEnvMultiAgent import Slime

import json
import numpy as np
import random
import datetime

PARAMS_FILE = "multi-agent-params.json"
TRAIN_EPISODES = 500
TRAIN_LOG_EVERY = 50
TEST_EPISODES = 10
TEST_LOG_EVERY = 1
OUTPUT_FILE = f"multi-test-01-{datetime.now()}.csv"

with open(PARAMS_FILE) as f:
    params = json.load(f)
env = Slime(render_mode="human", **params)

# Q-Learning
alpha = 0.2  # DOC learning rate (0 learn nothing 1 learn suddenly)
gamma = 0.8  # DOC discount factor (0 care only bout immediate rewards, 1 care only about future ones)
epsilon = 0.9  # DOC chance of random action
decay = 0.9995  # DOC di quanto diminuisce epsilon ogni episode

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

# Q_table
qtable = {i: np.zeros([4, 3]) for i in range(params['population'], params['population'] + params['learner_population'])}

# dict che tiene conto della frequenza di scelta delle action di ogni agent per ogni episodio {episode: {agent: {action: _, action: _, ...}}}
action_dict = {str(ep): {str(ag): {str(ac): 0 for ac in range(3)} for ag in range(params['population'], params['population']+params['learner_population'])} for ep in range(1, TRAIN_EPISODES+1)}
# dict che tiene conto della reward di ogni agente per ogni episodio {episode: {agent: _}}
reward_dict = {str(ep): {str(ag): 0 for ag in range(params['population'], params['population']+params['learner_population'])} for ep in range(1, TRAIN_EPISODES+1)}
# dict che tiene conto dela dimensioni di ogni cluster per ogni episodio
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
    env.reset()
    for tick in range(params['episode_ticks']):
        for agent in env.agent_iter(max_iter=params['learner_population']):
            state, _, _, _ = env.last(agent)
            s = state_to_int_map(state.observe())

            if random.uniform(0, 1) < epsilon:
                #action = np.random.randint(0, 2)
                action = env.action_space(agent).sample()
            else:
                action = np.argmax(qtable[agent][s])

            env.step(action)
            next_state, reward, _, _ = env.last(agent)  # get observation (state) for current agent
            next_s = state_to_int_map(next_state.observe())

            old_value = qtable[agent][s][action]
            next_max = np.max(qtable[agent][next_s])  # QUESTION: was with [action] too

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            qtable[agent][s][action] = new_value

            s = next_s

            action_dict[str(ep)][str(agent)][str(action)] += 1
            reward_dict[str(ep)][str(agent)] += round(reward, 2)
        env.move()
        env._evaporate()
        env._diffuse()
        env.render()
    epsilon *= decay
    if ep % TRAIN_LOG_EVERY == 0:
        print(f"EPISODE: {ep}")
        print(f"\tepsilon: {epsilon}")
        #print(f"\tepisode reward: {reward_episode}")
        avg_rew = 0
        for a in reward_dict[str(ep)]:
            avg_rew += reward_dict[str(ep)][a]
        avg_rew /= len(reward_dict[str(ep)])
        with open(OUTPUT_FILE, 'a') as f:
            f.write(f"{ep}, {round(avg_rew, 2)}\n")
    cluster_dict[str(ep)] = round(env.avg_cluster(), 2)
print(json.dumps(cluster_dict, indent=2))
print("Training finished!\n")

# DOC Evaluate agent's performance after Q-learning
cluster_dict = {}
print("Start testing...")
for ep in range(1, TEST_EPISODES+1):
    env.reset()
    for tick in range(params['episode_ticks']):
        for agent in env.agent_iter(max_iter=params['learner_population']):
            state, _, _, _ = env.last(agent)
            s = state_to_int_map(state.observe())

            if random.uniform(0, 1) < epsilon:
                # action = np.random.randint(0, 2)
                action = env.action_space(agent).sample()
            else:
                action = np.argmax(qtable[agent][s])

            env.step(action)
        env.move()
        env._evaporate()
        env._diffuse()
        env.render()
    if ep % TRAIN_LOG_EVERY == 0:
        print(f"EPISODE: {ep}")
        print(f"\tepsilon: {epsilon}")
        # print(f"\tepisode reward: {reward_episode}")
    cluster_dict[str(ep)] = round(env.avg_cluster(), 2)
print(json.dumps(cluster_dict, indent=2))
print("Testing finished!\n")
env.close()
