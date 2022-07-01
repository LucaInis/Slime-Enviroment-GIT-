import gym
import pygame
from gym import spaces
import numpy as np
import random


# creo un custom space
class Boolean(gym.Space):
    def __init__(self, size=None):
        assert isinstance(size, int) and size > 0
        self.size = size
        gym.Space.__init__(self, (), bool)

    def sample(self):
        bool = [False for i in range(self.size)]
        return bool


width = height = 400  # SUPPONGO CHE LA GRIGLIA SIA SEMPRE UN QUADRATO

episodes = 5
ticks_per_episode = 500
# consigliabile almeno 500 tick_per_episode, altrimenti difficile vedere fenomeni di aggregazione


class Slime(gym.Env):

    metadata = {"render_modes": "human", "render_fps": 30}

    # cluster_limit = cluster_threshold
    def __init__(self, sniff_threshold=10, step=5, cluster_limit=6, population=700, size_turle=2):
        self.size_turtle = size_turle
        self.sniff_threshold = sniff_threshold
        self.reward = 0
        self.reward_list = []
        self.step = step  # di quanto si muovono le turtle ogni tick
        self.population = population
        self.count_ticks_cluster = 0  # conta i tick che la turtle passa in un cluster
        self.cluster_limit = cluster_limit  # numero min di turtle affinché si consideri cluster (in range 20)

        # create learner turtle
        self.cord_learner_turtle = [np.random.randint(10, width - 10) for i in range(2)]

        # create NON learner turtle
        self.cord_non_learner_turtle = {}
        for p in range(self.population):
            self.l = [np.random.randint(10, width-10) for i in range(2)]
            self.cord_non_learner_turtle[str(p)] = self.l

        # patches-own [chemical] - amount of pheromone in each patch
        self.chemicals_level = {}
        for x in range(width + 1):
            for y in range(height + 1):
                self.chemicals_level[str(x) + str(y)] = 0
        self.action_space = spaces.Discrete(3)
        self.observation_space = Boolean(size=2)
        self.observation = self.observation_space.sample()

    # step function
    def moving_turtle(self, action: int):
        # MOVING NON LEARNER SLIME
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
                for y in range(self.bonds[1], self.bonds[3]):  # SCORRO LE "PATCH" NELLE VICINANE CON UN r = 3
                    if self.chemicals_level[str(x) + str(y)] > self.max_lv:  # CERCO IL MAX VALORE DI FEROMONE NELLE VICINANZE E PRENDO LE SUE COORDINATE
                        self.max_lv = self.chemicals_level[str(x) + str(y)]
                        self.cord_max_lv.clear()
                        self.cord_max_lv = []
                        self.cord_max_lv.append(x)
                        self.cord_max_lv.append(y)
            if self.max_lv > self.sniff_threshold:
                if self.cord_max_lv[0] > self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] > self.cord_non_learner_turtle[turtle][1]:
                    # allora il punto si trova in alto a dx
                    self.cord_non_learner_turtle[turtle][0] += self.step
                    self.cord_non_learner_turtle[turtle][1] += self.step
                elif self.cord_max_lv[0] < self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] < self.cord_non_learner_turtle[turtle][1]:
                    # allora il punto si trova in basso a sx
                    self.cord_non_learner_turtle[turtle][0] -= self.step
                    self.cord_non_learner_turtle[turtle][1] -= self.step
                elif self.cord_max_lv[0] > self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] < self.cord_non_learner_turtle[turtle][1]:
                    # allora il punto si trova in basso a dx
                    self.cord_non_learner_turtle[turtle][0] += self.step
                    self.cord_non_learner_turtle[turtle][1] -= self.step
                elif self.cord_max_lv[0] < self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] > self.cord_non_learner_turtle[turtle][1]:
                    # allora il punto si trova in alto a sx
                    self.cord_non_learner_turtle[turtle][0] -= self.step
                    self.cord_non_learner_turtle[turtle][1] += self.step
                elif self.cord_max_lv[0] == self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] < self.cord_non_learner_turtle[turtle][1]:
                    # allora il punto si trova in basso sulla mia colonna
                    self.cord_non_learner_turtle[turtle][1] -= self.step
                elif self.cord_max_lv[0] == self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] > self.cord_non_learner_turtle[turtle][1]:
                    # allora il punto si trova in alto sulla mia colonna
                    self.cord_non_learner_turtle[turtle][1] += self.step
                elif self.cord_max_lv[0] > self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] == self.cord_non_learner_turtle[turtle][1]:
                    # allora il punto si trova alla mia dx
                    self.cord_non_learner_turtle[turtle][0] += self.step
                elif self.cord_max_lv[0] < self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] == self.cord_non_learner_turtle[turtle][1]:
                    # allora il punto si trova alla mia sx
                    self.cord_non_learner_turtle[turtle][0] -= self.step
                else:
                    pass  # allora il punto è dove mi trovo quindi stò fermo
            else:
                # RANDOM WALK
                act = np.random.randint(4)
                if act == 0:
                    self.cord_non_learner_turtle[turtle][0] += self.step
                    self.cord_non_learner_turtle[turtle][1] += self.step
                elif act == 1:
                    self.cord_non_learner_turtle[turtle][0] -= self.step
                    self.cord_non_learner_turtle[turtle][1] -= self.step
                elif act == 2:
                    self.cord_non_learner_turtle[turtle][0] -= self.step
                    self.cord_non_learner_turtle[turtle][1] += self.step
                else:
                    self.cord_non_learner_turtle[turtle][0] += self.step
                    self.cord_non_learner_turtle[turtle][1] -= self.step

            # DROP CHEMICALS
            for x in range(self.bonds[0], self.bonds[2]):
                for y in range(self.bonds[1], self.bonds[3]):
                    self.chemicals_level[str(x) + str(y)] += 2

            # PER EVITARE ESCANO DALLO SCHERMO
            if self.cord_non_learner_turtle[turtle][0] > width - 10:
                self.cord_non_learner_turtle[turtle][0] = np.random.randint(10, width - 10)
            elif self.cord_non_learner_turtle[turtle][0] < 10:
                self.cord_non_learner_turtle[turtle][0] = np.random.randint(10, width - 10)
            if self.cord_non_learner_turtle[turtle][1] > height - 10:
                self.cord_non_learner_turtle[turtle][1] = np.random.randint(10, width - 10)
            elif self.cord_non_learner_turtle[turtle][1] < 10:
                self.cord_non_learner_turtle[turtle][1] = np.random.randint(10, width - 10)

        # MOVING LEARNER SLIME
        self.max = 0
        self.cord_max = []
        self.limit = []
        self.limit.append(self.cord_learner_turtle[0] - 3)  # start_x
        self.limit.append(self.cord_learner_turtle[1] - 3)  # start_y
        self.limit.append(self.cord_learner_turtle[0] + 4)  # end_x
        self.limit.append(self.cord_learner_turtle[1] + 4)  # end_y
        for i in range(len(self.limit)):
            if self.limit[i] < 0:
                self.limit[i] = 0
            elif self.limit[i] > width:
                self.limit[i] = width
        if action == 0:  # RANDOM WALK
            a = np.random.randint(4)  # faccio si che si possa muovere solo in diagonale
            if a == 0:
                self.cord_learner_turtle[0] += self.step
                self.cord_learner_turtle[1] += self.step
            elif a == 1:
                self.cord_learner_turtle[0] -= self.step
                self.cord_learner_turtle[1] += self.step
            elif a == 2:
                self.cord_learner_turtle[0] += self.step
                self.cord_learner_turtle[1] -= self.step
            else:
                self.cord_learner_turtle[0] -= self.step
                self.cord_learner_turtle[1] -= self.step

            # Per evitare che lo Slime learner esca dallo schermo
            if self.cord_learner_turtle[0] > width - 10:
                self.cord_learner_turtle[0] = width - 15
            elif self.cord_learner_turtle[0] < 10:
                self.cord_learner_turtle[0] = 15
            if self.cord_learner_turtle[1] > height - 10:
                self.cord_learner_turtle[1] = height - 15
            elif self.cord_learner_turtle[1] < 10:
                self.cord_learner_turtle[1] = 15
        elif action == 1:  # DROP CHEMICALS
            for x in range(self.limit[0], self.limit[2]):
                for y in range(self.limit[1], self.limit[3]):
                    self.chemicals_level[str(x) + str(y)] += 2
        elif action == 2:  # CHASE MAX CHEMICAL
            for x in range(self.limit[0], self.limit[2]):
                for y in range(self.limit[1], self.limit[3]):
                    if self.chemicals_level[str(x) + str(y)] > self.max:  # CERCO IL MAX VALORE DI FEROMONE NELLE VICINANZE E PRENDO LE SUE COORDINATE
                        self.max = self.chemicals_level[str(x) + str(y)]
                        self.cord_max.append(x)
                        self.cord_max.append(y)
            if self.max > self.sniff_threshold:
                if self.cord_max[0] > self.cord_learner_turtle[0] and self.cord_max[1] > self.cord_learner_turtle[1]:
                    self.cord_learner_turtle[0] += self.step
                    self.cord_learner_turtle[1] += self.step
                elif self.cord_max[0] < self.cord_learner_turtle[0] and self.cord_max[1] < self.cord_learner_turtle[1]:
                    self.cord_learner_turtle[0] -= self.step
                    self.cord_learner_turtle[1] -= self.step
                elif self.cord_max[0] > self.cord_learner_turtle[0] and self.cord_max[1] < self.cord_learner_turtle[1]:
                    self.cord_learner_turtle[0] += self.step
                    self.cord_learner_turtle[1] -= self.step
                elif self.cord_max[0] < self.cord_learner_turtle[0] and self.cord_max[1] > self.cord_learner_turtle[1]:
                    self.cord_learner_turtle[0] -= self.step
                    self.cord_learner_turtle[1] += self.step
                elif self.cord_max[0] < self.cord_learner_turtle[0] and self.cord_max[1] == self.cord_learner_turtle[1]:
                    self.cord_learner_turtle[0] -= self.step
                elif self.cord_max[0] > self.cord_learner_turtle[0] and self.cord_max[1] == self.cord_learner_turtle[1]:
                    self.cord_learner_turtle[0] += self.step
                elif self.cord_max[0] == self.cord_learner_turtle[0] and self.cord_max[1] < self.cord_learner_turtle[1]:
                    self.cord_learner_turtle[1] -= self.step
                elif self.cord_max[0] == self.cord_learner_turtle[0] and self.cord_max[1] > self.cord_learner_turtle[1]:
                    self.cord_learner_turtle[1] += self.step
                else:
                    pass
            else:
                pass

        self.reward = Slime.rewardfunc7(self)  # <--reward function in uso

        # EVAPORATE CHEMICAL
        for patch in self.chemicals_level:
            if self.chemicals_level[patch] != 0:
                self.chemicals_level[patch] -= 0.9

        self.observation = Slime.get_obs(self)

        return self.observation, self.reward, False, {}

    def get_obs(self):
        # controllo la presenza di feromone o meno nella patch
        if self.chemicals_level[str(self.cord_learner_turtle[0]) + str(self.cord_learner_turtle[1])] != 0:
            self.observation[1] = True
        else:
            self.observation[1] = False
        # controllo if in cluster
        self.count_turtle = 1
        self.check_cord = []
        for x in range(self.cord_learner_turtle[0] - 9, self.cord_learner_turtle[0] + 10):
            for y in range(self.cord_learner_turtle[1] - 9, self.cord_learner_turtle[1] + 10):
                self.check_cord.append([x, y])
        for pair in self.cord_non_learner_turtle.values():
            if pair in self.check_cord:
                self.count_turtle += 1
        if self.count_turtle >= self.cluster_limit:
            self.observation[0] = True
        else:
            self.observation[0] = False
        return self.observation

    def rewardfunc7(self):
        # ho semplificato di molto la reward function per evitare comportamenti statici
        obs = Slime.get_obs(self)
        if obs[0] == True:
            self.count_ticks_cluster += 1
            self.reward = 10 * self.count_ticks_cluster   # assegno la reward
            self.reward_list.append(self.reward)
        else:
            self.count_ticks_cluster = 0
            self.reward = -5  # assegno una penalty perché NON si trova in un cluster
            self.reward_list.append(self.reward)

        return self.reward

    def reset(self):
        self.reward = 0
        self.reward_list = []
        self.observation = self.observation_space.sample()
        self.count_ticks_cluster = 0

        # create learner turtle
        self.cord_learner_turtle = []
        self.cord_learner_turtle.append(np.random.randint(10, width - 10))
        self.cord_learner_turtle.append(np.random.randint(10, height - 10))

        # create NON learner turtle
        self.cord_non_learner_turtle = {}
        for p in range(self.population):
            self.l = []
            self.l.append(np.random.randint(10, width - 10))
            self.l.append(np.random.randint(10, height - 10))
            self.cord_non_learner_turtle[str(p)] = self.l

        # patches-own [chemical] - amount of pheromone in the patch
        self.chemicals_level = {}
        for x in range(width + 1):
            for y in range(height + 1):
                self.chemicals_level[str(x) + str(y)] = 0
        return self.observation, self.reward, False, {}

    def render(self):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("SLIME")
        self.screen.fill((0, 0, 0))
        self.clock = pygame.time.Clock()
        self.clock.tick(self.metadata["render_fps"])

        # Disegno LA turtle learner!
        pygame.draw.circle(self.screen, (190, 0, 0), (self.cord_learner_turtle[0], self.cord_learner_turtle[1]), self.size_turtle)

        for turtle in self.cord_non_learner_turtle.values():
            pygame.draw.circle(self.screen, (0, 190, 0), (turtle[0], turtle[1]), self.size_turtle)
        pygame.display.flip()

    def close(self):
        pygame.display.quit()
        pygame.quit()


#   MAIN


env = Slime()

# Q-Learning
alpha = 0.1
gamma = 0.6
epsilon = 0.9
decay = 0.95  # di quanto diminuisce epsilon ogni episode


q_table = np.zeros([4, env.action_space.n])


# NB: la fase di training dura circa 10 minuti con 16 episodi
# TRAINING
print("Start training...")
for ep in range(1, 16):
    print(f"EPISODE: {ep}")
    print("Epsilon: ", epsilon)
    state, reward, done, info = env.reset()
    if sum(state) == 0:     # [False, False]
        state = sum(state)      # 0
    elif sum(state) == 2:       # [True, True]
        state = 3
    elif int(state[0]) == 1 and int(state[1]) == 0:     # [True, False] ==> si trova in un cluster ma non su una patch con feromone --> difficile succeda
        state = 1
    else:
        state = 2       # [False, True]
    for tick in range(500):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, info = env.moving_turtle(action)
        if sum(next_state) == 0:  # [False, False]
            next_state = sum(next_state)  # 0
        elif sum(next_state) == 2:  # [True, True]
            next_state = 3
        elif int(next_state[0]) == 1 and int(next_state[1]) == 0:  # [True, False]
            next_state = 1
        else:
            next_state = 2  # [False, True]

        old_value = q_table[state][action]
        next_max = np.max(q_table[state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state][action] = new_value

        state = next_state
    epsilon *= decay
    print(q_table)
print("Training finished!\n")


"""Evaluate agent's performance after Q-learning"""
for ep in range(1, episodes+1):
    reward_episode = 0
    state, reward, done, info = env.reset()
    state = sum(state)
    print(f"EPISODE: {ep}")
    for tick in range(ticks_per_episode):
        action = np.argmax(q_table[state])
        state, reward, done, info = env.moving_turtle(action)
        state = sum(state)
        reward_episode += reward
        env.render()
env.close()

print("END")
