import gym
import pygame
from gym import spaces
import numpy as np


# creo un custom space
class Boolean(gym.Space):
    def __init__(self, size=None):
        assert isinstance(size, int) and size > 0
        self.size = size
        gym.Space.__init__(self, (), bool)

    def sample(self):
        bool = [random.choice([True, False]) for i in range(self.size)]
        return bool


width = height = 500  # SUPPONGO CHE LA GRIGLIA SIA SEMPRE UN QUADRATO

episodes = 5
ticks_per_episode = 700
# consigliabile almeno 500 tick_per_episode, altrimenti difficile vedere fenomeni di aggregazione


class Slime(gym.Env):

    metadata = {"render_modes": "human", "render_fps": 30}

    # cluster_limit = cluster_threshold
    def __init__(self, sniff_threshold=12, step=5, cluster_limit=5, population=650):
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
        self.observation = [False, False]

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
                self.cord_non_learner_turtle[turtle][0] = width - 15
            elif self.cord_non_learner_turtle[turtle][0] < 10:
                self.cord_non_learner_turtle[turtle][0] = 15
            if self.cord_non_learner_turtle[turtle][1] > height - 10:
                self.cord_non_learner_turtle[turtle][1] = height - 15
            elif self.cord_non_learner_turtle[turtle][1] < 10:
                self.cord_non_learner_turtle[turtle][1] = 15

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
                self.chemicals_level[patch] -= 2

        self.observation = Slime.get_obs(self)

        return self.observation, self.reward, False, {}

    def get_obs(self):
        # controllo la presenza di feromone o meno nella patch, da spostare
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

    def rewardfunc1(self):
        self.count_turtle = 1
        self.check_cord = []
        for x in range(self.cord_learner_turtle[0] - 9, self.cord_learner_turtle[0] + 10):
            for y in range(self.cord_learner_turtle[1] - 9, self.cord_learner_turtle[1] + 10):
                self.check_cord.append([x, y])
        for pair in self.cord_non_learner_turtle.values():
            if pair in self.check_cord:
                self.count_turtle += 1
        if self.count_turtle >= self.cluster_limit:
            self.count_ticks_cluster += 1
            self.reward = 2
            self.reward_list.append(2)   # se la mia turtle è in un cluster gli assegno una reward
        else:
            self.count_ticks_cluster = 0
            self.reward = -2
            self.reward_list.append(-0.2)  # se la mia turtle NON è in un cluster gli assegno una penalty

        return self.reward

    def rewardfunc2(self):
        self.count_turtle = 1
        self.check_cord = []
        for x in range(coordinate[0] - 9, coordinate[0] + 10):
            for y in range(coordinate[1] - 9, coordinate[1] + 10):
                self.check_cord.append([x, y])
        for pair in self.cord_non_learner_turtle.values():
            if pair in self.check_cord:
                self.count_turtle += 1
        if self.count_turtle >= self.cluster_limit:
            self.count_ticks_cluster += 1
        else:
            self.count_ticks_cluster = 0
        if self.count_ticks_cluster > 1:
            self.reward_list.append(self.count_ticks_cluster)  # monotonic reward based on ticks in cluster

        return self.count_ticks_cluster

    def rewardfunc7(self):
        self.reward = 0
        self.count_turtle = 1
        self.check_cord = []
        for x in range(self.cord_learner_turtle[0] - 9, self.cord_learner_turtle[0] + 10):
            for y in range(self.cord_learner_turtle[1] - 9, self.cord_learner_turtle[1] + 10):
                self.check_cord.append([x, y])
        for pair in self.cord_non_learner_turtle.values():
            if pair in self.check_cord:
                self.count_turtle += 1
        if self.count_turtle >= self.cluster_limit:
            self.count_ticks_cluster += 1
            # calcolo la reward
            self.reward = ((self.count_turtle ^ 2) / self.cluster_limit) + (
                        (ticks_per_episode - self.count_ticks_cluster) / ticks_per_episode)
            self.reward_list.append(self.reward)
        else:
            self.reward = -0.5  # assegno una penalty perché NON si trova in un cluster
            self.reward_list.append(self.reward)
            self.count_ticks_cluster = 0

        return self.reward

    def reset(self):
        self.reward = 0
        self.reward_list = []
        self.observation = [False, False]
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
        pygame.draw.circle(self.screen, (190, 0, 0), (self.cord_learner_turtle[0], self.cord_learner_turtle[1]), 3)

        for turtle in self.cord_non_learner_turtle.values():
            pygame.draw.circle(self.screen, (0, 190, 0), (turtle[0], turtle[1]), 3)
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


#   MAIN


env = Slime()
for ep in range(1, episodes+1):
    env.reset()
    print(f"EPISODE: {ep}")
    for tick in range(ticks_per_episode):
        observation, reward, done, info = env.moving_turtle(env.action_space.sample())
        # if tick % 2 == 0:
        print(observation, reward)
        env.render()
env.close()
