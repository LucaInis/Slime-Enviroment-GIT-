import gym
from gym.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import numpy as np
from gym.spaces import Discrete
import pygame
import random

class Boolean(gym.Space):
    def __init__(self, size=None):
        assert isinstance(size, int) and size > 0
        self.size = size
        gym.Space.__init__(self, (), bool)

    def sample(self):
        # bool = [random.choice([True, False]) for i in range(self.size)]
        bool = [False for i in range(self.size)]
        return bool


width = height = 200  # dimensioni pop up pygame

learner_population = 3
episodes = 2
ticks_per_episode = 800


class Slime(AECEnv):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, population=300, sniff_threshold=20, cluster_limit=5, step_t=2, turtle_size=1):
        self.seed()

        self.agent_list = []
        self.agents = []

        self.turtle_size = turtle_size

        # create le learner turtle
        self.learner_population = learner_population
        self.agent_coordinate = {}
        for i in range(self.learner_population):
            agent_name = "Turtle_" + str(i)
            self.agents.append(agent_name)  # list of all agents
            # agent_name_mapping
            self.agent_coordinate[agent_name] = [np.random.randint(10, width - 10) for i in range(2)]

        # inizializzo il selettore
        self._agent_selector = agent_selector(self.agents)

        self.rewards = {a: 0 for a in self.agents}  # reward from the last step for each agent
        self._cumulative_rewards_ = {a: 0 for a in self.agents}  # cumulative rewards for each agent
        self.count_ticks_cluster = {a: 0 for a in self.agents}

        self.action_spaces = dict(zip(self.agents, [Discrete(3) for a in enumerate(self.agents)]))
        self.observation_spaces = Boolean(size=2)
        self.obs_dict = {a: self.observation_spaces.sample() for a in self.agents}
        self.sniff_threshold = sniff_threshold
        self.step_t = step_t
        self.cluster_limit = cluster_limit

        # create NON learner turtle
        self.population = population
        self.cord_non_learner_turtle = {}
        for p in range(self.population):
            self.l = [np.random.randint(10, width - 10) for i in range(2)]
            self.cord_non_learner_turtle[str(p)] = self.l

        # inizializzo il valore di feromone nella griglia
        self.chemicals_level = {}
        for x in range(width + 1):
            for y in range(height + 1):
                self.chemicals_level[str(x) + str(y)] = 0

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        self.check_cord = []
        self.count_turtle = 1  # perché c'é di sicuro già la learner turtle
        for x in range(self.agent_coordinate[agent][0] - 9, self.agent_coordinate[agent][0] + 10):
            for y in range(self.agent_coordinate[agent][1] - 9, self.agent_coordinate[agent][1] + 10):
                self.check_cord.append([x, y])
        for pair in self.cord_non_learner_turtle.values():
            if pair in self.check_cord:
                self.count_turtle += 1
        if self.count_turtle >= self.cluster_limit:
            self.obs_dict[agent][0] = True
        else:
            self.obs_dict[agent][0] = False
        self.check_cord.clear()

        if self.chemicals_level[str(self.agent_coordinate[agent][0]) + str(self.agent_coordinate[agent][1])] > 0: ########
            self.obs_dict[agent][1] = True
        else:
            self.obs_dict[agent][1] = False

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed=seed)

        self.agent_list = []
        self.agents = []

        # create NON learner turtle
        self.cord_non_learner_turtle = {}
        for p in range(self.population):
            self.l = [np.random.randint(10, width - 10) for i in range(2)]
            self.cord_non_learner_turtle[str(p)] = self.l

            # create le learner turtle
            self.learner_population = learner_population
            self.agent_coordinate = {}
            for i in range(self.learner_population):
                agent_name = "Turtle_" + str(i)
                self.agents.append(agent_name)  # list of all agents
                # agent_name_mapping
                self.agent_coordinate[agent_name] = [np.random.randint(10, width - 10) for i in range(2)]

        self.count_ticks_cluster = {a: 0 for a in self.agents}
        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards_ = {a: 0 for a in self.agents}

        self.chemicals_level = {}
        for x in range(width + 1):
            for y in range(height + 1):
                self.chemicals_level[str(x) + str(y)] = 0

        self.obs_dict = {a: self.observation_spaces.sample() for a in self.agents}

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    # moving NON learner turtle
    def moving(self):
        for turtle in self.cord_non_learner_turtle:
            self.max_lv = 0
            self.cord_max_lv = []
            self.bonds = []
            self.bonds.append(self.cord_non_learner_turtle[turtle][0] - 3)
            self.bonds.append(self.cord_non_learner_turtle[turtle][1] - 3)
            self.bonds.append(self.cord_non_learner_turtle[turtle][0] + 4)
            self.bonds.append(self.cord_non_learner_turtle[turtle][1] + 4)
            for i in range(len(self.bonds)):
                if self.bonds[i] < 0:
                    self.bonds[i] = 0
                elif self.bonds[i] > width:
                    self.bonds[i] = width
            for x in range(self.bonds[0], self.bonds[2]):
                for y in range(self.bonds[1], self.bonds[3]):  # scorro le "patch" vicine (r = 3)
                    # cerco il max valore di feromone nelle vicinanze
                    if self.chemicals_level[str(x) + str(y)] > self.max_lv:
                        self.max_lv = self.chemicals_level[str(x) + str(y)]
                        self.cord_max_lv.clear()
                        self.cord_max_lv = []
                        self.cord_max_lv.append(x)
                        self.cord_max_lv.append(y)
            if self.max_lv > self.sniff_threshold:
                if self.cord_max_lv[0] > self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] > \
                        self.cord_non_learner_turtle[turtle][1]:
                    # allora il punto si trova in alto a dx
                    self.cord_non_learner_turtle[turtle][0] += self.step_t
                    self.cord_non_learner_turtle[turtle][1] += self.step_t
                elif self.cord_max_lv[0] < self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] < \
                        self.cord_non_learner_turtle[turtle][1]:
                    # allora il punto si trova in basso a sx
                    self.cord_non_learner_turtle[turtle][0] -= self.step_t
                    self.cord_non_learner_turtle[turtle][1] -= self.step_t
                elif self.cord_max_lv[0] > self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] < \
                        self.cord_non_learner_turtle[turtle][1]:
                    # allora il punto si trova in basso a dx
                    self.cord_non_learner_turtle[turtle][0] += self.step_t
                    self.cord_non_learner_turtle[turtle][1] -= self.step_t
                elif self.cord_max_lv[0] < self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] > \
                        self.cord_non_learner_turtle[turtle][1]:
                    # allora il punto si trova in alto a sx
                    self.cord_non_learner_turtle[turtle][0] -= self.step_t
                    self.cord_non_learner_turtle[turtle][1] += self.step_t
                elif self.cord_max_lv[0] == self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] < \
                        self.cord_non_learner_turtle[turtle][1]:
                    # allora il punto si trova in basso sulla mia colonna
                    self.cord_non_learner_turtle[turtle][1] -= self.step_t
                elif self.cord_max_lv[0] == self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] > \
                        self.cord_non_learner_turtle[turtle][1]:
                    # allora il punto si trova in alto sulla mia colonna
                    self.cord_non_learner_turtle[turtle][1] += self.step_t
                elif self.cord_max_lv[0] > self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] == \
                        self.cord_non_learner_turtle[turtle][1]:
                    # allora il punto si trova alla mia dx
                    self.cord_non_learner_turtle[turtle][0] += self.step_t
                elif self.cord_max_lv[0] < self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] == \
                        self.cord_non_learner_turtle[turtle][1]:
                    # allora il punto si trova alla mia sx
                    self.cord_non_learner_turtle[turtle][0] -= self.step_t
                else:
                    pass  # allora il punto è dove mi trovo
            else:
                # RANDOM WALK
                act = np.random.randint(4)
                if act == 0:
                    self.cord_non_learner_turtle[turtle][0] += self.step_t
                    self.cord_non_learner_turtle[turtle][1] += self.step_t
                elif act == 1:
                    self.cord_non_learner_turtle[turtle][0] -= self.step_t
                    self.cord_non_learner_turtle[turtle][1] -= self.step_t
                elif act == 2:
                    self.cord_non_learner_turtle[turtle][0] -= self.step_t
                    self.cord_non_learner_turtle[turtle][1] += self.step_t
                else:
                    self.cord_non_learner_turtle[turtle][0] += self.step_t
                    self.cord_non_learner_turtle[turtle][1] -= self.step_t

            # DROP CHEMICALS
            for x in range(self.bonds[0], self.bonds[2]):
                for y in range(self.bonds[1], self.bonds[3]):
                    self.chemicals_level[str(x) + str(y)] += 2

            # per evitare escano dallo schermo
            if self.cord_non_learner_turtle[turtle][0] > width - 10:
                self.cord_non_learner_turtle[turtle][0] = width - 15
            elif self.cord_non_learner_turtle[turtle][0] < 10:
                self.cord_non_learner_turtle[turtle][0] = 15
            if self.cord_non_learner_turtle[turtle][1] > height - 10:
                self.cord_non_learner_turtle[turtle][1] = height - 15
            elif self.cord_non_learner_turtle[turtle][1] < 10:
                self.cord_non_learner_turtle[turtle][1] = 15

    def evaporate_chemical(self):
        for patch in self.chemicals_level:
            if self.chemicals_level[patch] != 0:
                self.chemicals_level[patch] -= 2

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def rewardFunction7(self, ag):
        self.rew = 0
        self.count_turtle = 1  # tiene conto di quante turtle ci sono "intorno", parte da 1 perché c'é di sicuro ag
        self.check_cord = []
        for x in range(self.agent_coordinate[ag][0] - 9, self.agent_coordinate[ag][0] + 10):
            for y in range(self.agent_coordinate[ag][1] - 9, self.agent_coordinate[ag][1] + 10):
                self.check_cord.append([x, y])
        for pair in self.cord_non_learner_turtle.values():
            if pair in self.check_cord:
                self.count_turtle += 1
        if self.count_turtle >= self.cluster_limit:  # allora vuol dire che è in un cluster
            self.count_ticks_cluster[ag] += 1
            # calcolo la reward
            self.rew = 10
        else:
            self.rew = -0.5  # assegno una penalty perché NON si trova in un cluster
            self.count_ticks_cluster[ag] = 0
        return self.rew

    # funzione che uso per scorrere la lista degli agenti
    def flow_agent(self,agent):
        agent = self.agent_selection
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        agent = self.agent_selection  # selezioni l'agente
        self.max = 0
        self.cord_max = []
        self.limit = []
        self.limit.append(self.agent_coordinate[agent][0] - 3)  # start_x
        self.limit.append(self.agent_coordinate[agent][1] - 3)  # start_y
        self.limit.append(self.agent_coordinate[agent][0] + 4)  # end_x
        self.limit.append(self.agent_coordinate[agent][1] + 4)  # end_y
        for i in range(len(self.limit)):
            if self.limit[i] < 0:
                self.limit[i] = 0
            elif self.limit[i] > width:
                self.limit[i] = width
        if action == 0:  # RANDOM WALK
            a = np.random.randint(4)  # faccio si che si possa muovere solo in diagonale
            if a == 0:
                self.agent_coordinate[agent][0] += self.step_t
                self.agent_coordinate[agent][1] += self.step_t
            elif a == 1:
                self.agent_coordinate[agent][0] -= self.step_t
                self.agent_coordinate[agent][1] += self.step_t
            elif a == 2:
                self.agent_coordinate[agent][0] += self.step_t
                self.agent_coordinate[agent][1] -= self.step_t
            else:
                self.agent_coordinate[agent][0] -= self.step_t
                self.agent_coordinate[agent][1] -= self.step_t

            # Per evitare che lo Slime learner esca dallo schermo
            if self.agent_coordinate[agent][0] > width - 10:
                self.agent_coordinate[agent][0] = width - 15
            elif self.agent_coordinate[agent][0] < 10:
                self.agent_coordinate[agent][0] = 15
            if self.agent_coordinate[agent][1] > height - 10:
                self.agent_coordinate[agent][1] = height - 15
            elif self.agent_coordinate[agent][1] < 10:
                self.agent_coordinate[agent][1] = 15
        elif action == 1:  # DROP CHEMICALS
            for x in range(self.limit[0], self.limit[2]):
                for y in range(self.limit[1], self.limit[3]):
                    self.chemicals_level[str(x) + str(y)] += 2
            for x in range(self.limit[0]+2, self.limit[2]-2):
                for y in range(self.limit[1]+2, self.limit[3]-2):
                    self.chemicals_level[str(x) + str(y)] += 2
        elif action == 2:  # CHASE MAX CHEMICAL
            for x in range(self.limit[0], self.limit[2]):
                for y in range(self.limit[1], self.limit[3]):
                    if self.chemicals_level[str(x) + str(
                            y)] > self.max:  # CERCO IL MAX VALORE DI FEROMONE NELLE VICINANZE E PRENDO LE SUE COORDINATE
                        self.max = self.chemicals_level[str(x) + str(y)]
                        self.cord_max.append(x)
                        self.cord_max.append(y)
            if self.max > self.sniff_threshold:
                if self.cord_max[0] > self.agent_coordinate[agent][0] and self.cord_max[1] > self.agent_coordinate[agent][1]:
                    self.agent_coordinate[agent][0] += self.step_t
                    self.agent_coordinate[agent][1] += self.step_t
                elif self.cord_max[0] < self.agent_coordinate[agent][0] and self.cord_max[1] < self.agent_coordinate[agent][1]:
                    self.agent_coordinate[agent][0] -= self.step_t
                    self.agent_coordinate[agent][1] -= self.step_t
                elif self.cord_max[0] > self.agent_coordinate[agent][0] and self.cord_max[1] < self.agent_coordinate[agent][1]:
                    self.agent_coordinate[agent][0] += self.step_t
                    self.agent_coordinate[agent][1] -= self.step_t
                elif self.cord_max[0] < self.agent_coordinate[agent][0] and self.cord_max[1] > self.agent_coordinate[agent][1]:
                    self.agent_coordinate[agent][0] -= self.step_t
                    self.agent_coordinate[agent][1] += self.step_t
                elif self.cord_max[0] < self.agent_coordinate[agent][0] and self.cord_max[1] == self.agent_coordinate[agent][1]:
                    self.agent_coordinate[agent][0] -= self.step_t
                elif self.cord_max[0] > self.agent_coordinate[agent][0] and self.cord_max[1] == self.agent_coordinate[agent][1]:
                    self.agent_coordinate[agent][0] += self.step_t
                elif self.cord_max[0] == self.agent_coordinate[agent][0] and self.cord_max[1] < self.agent_coordinate[agent][1]:
                    self.agent_coordinate[agent][1] -= self.step_t
                elif self.cord_max[0] == self.agent_coordinate[agent][0] and self.cord_max[1] > self.agent_coordinate[agent][1]:
                    self.agent_coordinate[agent][1] += self.step_t
                else:
                    pass
            else:
                pass
            # Per evitare che lo Slime learner esca dallo schermo
            if self.agent_coordinate[agent][0] > width - 10:
                self.agent_coordinate[agent][0] = width - 15
            elif self.agent_coordinate[agent][0] < 10:
                self.agent_coordinate[agent][0] = 15
            if self.agent_coordinate[agent][1] > height - 10:
                self.agent_coordinate[agent][1] = height - 15
            elif self.agent_coordinate[agent][1] < 10:
                self.agent_coordinate[agent][1] = 15

        self.agent_selection = self._agent_selector.next()  # passo al prossimo agente

    def last(self, agent):
        Slime.observe(self, agent)  # calcola le observation
        self.reward = Slime.rewardFunction7(self, agent)  # <--reward function in uso
        self.rewards[agent] = self.reward
        self._cumulative_rewards_[agent] += self.reward
        return self.obs_dict[agent], self.rewards[agent], False, "info"

    def render(self, mode="human"):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("TURTLE")
        self.screen.fill((0, 0, 0))
        self.clock = pygame.time.Clock()
        self.clock.tick(self.metadata["render_fps"])

        # Disegno learner turtle
        for turtle in self.agent_coordinate.values():
            pygame.draw.circle(self.screen, (170, 0, 0), (turtle[0], turtle[1]), self.turtle_size)

        # Disegno NON learner turtle
        for turtle in self.cord_non_learner_turtle.values():
            pygame.draw.circle(self.screen, (0, 170, 0), (turtle[0], turtle[1]), self.turtle_size)
        pygame.display.flip()


# MAIN

env = Slime()

# Q-Learning
alpha = 0.1
gamma = 0.6
epsilon = 0.9
decay = 0.95  # di quanto diminuisce epsilon ogni episode

# scorro la lista degli agenti e per ognuno creo (iniziallizzo) un a Q_table
d_qtable = {}
env.reset()
for a in env.agent_iter(max_iter=learner_population):
    env.flow_agent(a)
    d_qtable[a] = np.zeros([4, 3])
#print(d_qtable)


# TRAINING
print("Start training...")
for ep in range(1, 16):
    print(f"EPISODE: {ep}")
    print("Epsilon: ", epsilon)
    env.reset()
    for tick in range(500):
        env.moving()
        for agent in env.agent_iter(max_iter=learner_population):
            state, reward, done, info = env.last(agent)  # get observation (state) for current agent

            if sum(state) == 0:  # [False, False]
                state = sum(state)  # 0
            elif sum(state) == 2:  # [True, True]
                state = 3
            elif int(state[0]) == 1 and int(state[1]) == 0:  # [True, False] ==> si trova in un cluster ma non su una patch con feromone --> difficile succeda
                state = 1
            else:
                state = 2  # [False, True]

            if random.uniform(0, 1) < epsilon:
                action = env.action_space(agent).sample()  # Explore action space
            else:
                action = np.argmax(d_qtable[agent][state])  # Exploit learned values

            env.step(action)
            next_state, reward, done, info = env.last(agent)  # get observation (state) for current agent
            if sum(next_state) == 0:  # [False, False]
                next_state = sum(next_state)  # 0
            elif sum(next_state) == 2:  # [True, True]
                next_state = 3
            elif int(next_state[0]) == 1 and int(next_state[1]) == 0:  # [True, False]
                next_state = 1
            else:
                next_state = 2  # [False, True]
            old_value = d_qtable[agent][state][action]

            next_max = np.max(d_qtable[agent][next_state][action])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            d_qtable[agent][state][action] = new_value

            state = next_state

    epsilon *= decay
print("Training finished!\n")



"""Evaluate agent's performance after Q-learning"""
for ep in range(1, episodes+1):
    env.reset()
    print(f"Episode: {ep}")
    for tick in range(ticks_per_episode):
        env.moving()
        for agent in env.agent_iter(max_iter=learner_population):
            observation, reward, done, info = env.last(agent)
            #print(observation)
            action = np.argmax(d_qtable[agent][state])
            env.step(action)
        env.render()
        env.evaporate_chemical()
env.close()

