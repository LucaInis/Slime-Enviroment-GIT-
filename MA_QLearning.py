from SlimeEnvMultiAgent import Slime

import json
import numpy as np
import random
import datetime

PARAMS_FILE = "multi-agent-params.json"
LEARNING_PARAMS_FILE = "ma-learning-params.json"
with open(LEARNING_PARAMS_FILE) as f:
    l_params = json.load(f)
OUTPUT_FILE = f"{l_params['OUTPUT_FILE']}-{datetime.datetime.now()}.csv"
with open(PARAMS_FILE) as f:
    params = json.load(f)
env = Slime(render_mode="human", **params)

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
    for l in range(params['population'], params['population']+params['learner_population']):
        for a in l_params["actions"]:
            f.write(f"(learner {l})-{a}, ")
    f.write("Avg reward X episode\n")

# Q_table
qtable = {i: np.zeros([4, 3]) for i in range(params['population'], params['population'] + params['learner_population'])}

# DOC dict che tiene conto della frequenza di scelta delle action per ogni episodio {episode: {action: _, action: _, ...}}
actions_dict = {str(ep): {str(ac): 0 for ac in range(3)} for ep in range(1, TRAIN_EPISODES+1)}  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
# DOC dict che tiene conto della frequenza di scelta delle action di ogni agent per ogni episodio {episode: {agent: {action: _, action: _, ...}}}
action_dict = {str(ep): {str(ag): {str(ac): 0 for ac in range(3)} for ag in range(params['population'], params['population']+params['learner_population'])} for ep in range(1, TRAIN_EPISODES+1)}
# DOC dict che tiene conto della reward di ogni agente per ogni episodio {episode: {agent: _}}
reward_dict = {str(ep): {str(ag): 0 for ag in range(params['population'], params['population']+params['learner_population'])} for ep in range(1, TRAIN_EPISODES+1)}
# DOC dict che tiene conto dela dimensioni di ogni cluster per ogni episodio
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
old_s = {}  # DOC old state for each agent {agent: old_state}
for ep in range(1, TRAIN_EPISODES+1):
    env.reset()
    for tick in range(1, params['episode_ticks']+1):
        for agent in env.agent_iter(max_iter=params['learner_population']):
            cur_state, reward, _, _ = env.last(agent)
            cur_s = state_to_int_map(cur_state.observe())
            if ep == 1 and tick == 1:
                action = env.action_space(agent).sample()
            else:
                old_value = qtable[agent][old_s[agent]][action]
                next_max = np.max(qtable[agent][cur_s])  # QUESTION: was with [action] too
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                qtable[agent][old_s[agent]][action] = new_value

                if random.uniform(0, 1) < epsilon:
                    # action = np.random.randint(0, 2)
                    action = env.action_space(agent).sample()
                else:
                    action = np.argmax(qtable[agent][cur_s])
            env.step(action)

            old_s[agent] = cur_s

            actions_dict[str(ep)][str(action)] += 1
            action_dict[str(ep)][str(agent)][str(action)] += 1
            reward_dict[str(ep)][str(agent)] += round(reward, 2)
        env.move()
        env._evaporate()
        env._diffuse()
        env.render()
        #print(json.dumps(action_dict, indent=2))
    epsilon *= decay
    cluster_dict[str(ep)] = round(env.avg_cluster(), 2)
    if ep % TRAIN_LOG_EVERY == 0:
        print(f"EPISODE: {ep}")
        print(f"\tepsilon: {epsilon}")
        #print(f"\tepisode reward: {reward_episode}")
        # From NetlogoDataAnalysis: Episode, Tick, Avg cluster size X tick, move-toward-chemical (2), random-walk (0), drop-chemical (1), (learner 0)-move-toward-chemical, ..., Avg reward X episode
        with open(OUTPUT_FILE, 'a') as f:
            f.write(f"{ep}, {params['episode_ticks'] * ep}, {cluster_dict[str(ep)]}, {actions_dict[str(ep)]['2']}, {actions_dict[str(ep)]['0']}, {actions_dict[str(ep)]['1']}, ")
            avg_rew = 0
            for l in range(params['population'], params['population']+params['learner_population']):
                avg_rew += (reward_dict[str(ep)][str(l)] / params['episode_ticks'])
                f.write(f"{action_dict[str(ep)][str(l)]['2']}, {action_dict[str(ep)][str(l)]['0']}, {action_dict[str(ep)][str(l)]['1']}, ")
            avg_rew /= params['learner_population']
            f.write(f"{avg_rew}\n")

#print(json.dumps(cluster_dict, indent=2))
print("Training finished!\n")

# DOC Evaluate agent's performance after Q-learning
cluster_dict = {}
print("Start testing...")
for ep in range(1, TEST_EPISODES+1):
    env.reset()
    for tick in range(1, params['episode_ticks']+1):
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
    if ep % TEST_LOG_EVERY == 0:
        print(f"EPISODE: {ep}")
        print(f"\tepsilon: {epsilon}")
        # print(f"\tepisode reward: {reward_episode}")
    cluster_dict[str(ep)] = round(env.avg_cluster(), 2)
print(json.dumps(cluster_dict, indent=2))
print("Testing finished!\n")
env.close()
