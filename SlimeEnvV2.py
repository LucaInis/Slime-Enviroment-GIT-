from typing import Optional

import gym
import pygame
from gym import spaces

import numpy as np
import random


class BooleanSpace(gym.Space):  # TODO improve implementation: should be a N-dimensional space of N boolean values
    def __init__(self, size=None):
        assert isinstance(size, int) and size > 0
        self.size = size
        self.values = [False for _ in range(self.size)]
        gym.Space.__init__(self, (), bool)

    def contains(self, x):
        return x in self.values

    def sample(self):
        #return [random.choice([True, False]) for _ in range(self.size)]
        return self.values


class Slime(gym.Env):
    metadata = {"render_modes": "human", "render_fps": 30}

    def __init__(self,
                 population=650,
                 sniff_threshold=12,
                 cluster_threshold=5,
                 cluster_radius=20,
                 rew=100,
                 penalty=-1,
                 render_mode: Optional[str] = None,
                 step=5,
                 grid_size=500):
        """

        :param population: Controls the number of non-learning slimes (= green turtles)
        :param sniff_threshold: Controls how sensitive slimes are to pheromone (higher values make slimes less sensitive
        to pheromone)—unclear effect on learning, could be negligible
        :param cluster_threshold: Controls the minimum number of slimes needed to consider an aggregate within
        cluster-radius a cluster (the higher the more difficult to consider an aggregate a cluster)—the higher the more
        difficult to obtain a positive reward for being within a cluster for learning slimes
        :param cluster_radius: Controls the range considered by slimes to count other slimes within a cluster (the
        higher the easier to form clusters, as turtles far apart are still counted together)—the higher the easier it is
        to obtain a positive reward for being within a cluster for learning slimes
        :param rew: Base reward for being in a cluster
        :param penalty: Base penalty for not being in a cluster
        :param render_mode:
        :param step: How many pixels do turtle move at each movement step
        :param grid_size: Simulation area is always a square
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.population = population
        self.sniff_threshold = sniff_threshold
        self.cluster_threshold = cluster_threshold
        self.cluster_radius = cluster_radius
        self.reward = rew
        self.penalty = penalty

        self.move_step = step
        self.width = grid_size
        self.height = grid_size

        self.reward_list = []
        self.count_ticks_cluster = 0    # conta i tick che la turtle passa in un cluster

        self.first_gui = True

        # create learner turtle
        self.cord_learner_turtle = [np.random.randint(10, self.width - 10) for _ in range(2)]
        # create NON learner turtle
        self.cord_non_learner_turtle = {}
        for p in range(self.population):
            self.l = [np.random.randint(10, self.width-10) for _ in range(2)]
            self.cord_non_learner_turtle[str(p)] = self.l

        # patches-own [chemical] - amount of pheromone in each patch
        self.chemicals_level = {}
        for x in range(self.width + 1):
            for y in range(self.height + 1):
                self.chemicals_level[str(x) + str(y)] = 0

        self.action_space = spaces.Discrete(3)  # DOC 0 = rng_walk, 1 = drop_chemical, 2 = follow_pheromone TODO as dict
        self.observation_space = BooleanSpace(size=2)   # DOC observation_space[0] = whether there is chemical in turtle
                                                        # patch
                                                        # DOC observation_space[1] = whether the turtle is in a cluster
        self.observation = [False, False]   # FIXME di fatto non usi lo spazio in questo modo

    def step(self, action: int):
        """

        :param action:
        :return:
        """
        # MOVING NON LEARNER SLIME
        for turtle in self.cord_non_learner_turtle:
            self.max_lv = 0
            self.cord_max_lv = []
            self.bonds = []

            self._find_max_lv(turtle)  # DOC find patch where chemical is max

            if self.max_lv > self.sniff_threshold:

                self.follow_pheromone(turtle)  # DOC move towards greatest pheromone

            else:
                # RANDOM WALK
                self.rng_walk(turtle)

            # DROP CHEMICALS
            self.drop_chemical()

            # PER EVITARE ESCANO DALLO SCHERMO
            self._keep_in_screen(turtle)

        # FIXME codice quasi esattamente duplicato da find_max_lv()
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
            elif self.limit[i] > self.width:
                self.limit[i] = self.width
        # FIXME codice quasi esattamente duplicato da rng_walk()
        if action == 0:  # RANDOM WALK
            a = np.random.randint(4)  # faccio si che si possa muovere solo in diagonale
            if a == 0:
                self.cord_learner_turtle[0] += self.move_step
                self.cord_learner_turtle[1] += self.move_step
            elif a == 1:
                self.cord_learner_turtle[0] -= self.move_step
                self.cord_learner_turtle[1] += self.move_step
            elif a == 2:
                self.cord_learner_turtle[0] += self.move_step
                self.cord_learner_turtle[1] -= self.move_step
            else:
                self.cord_learner_turtle[0] -= self.move_step
                self.cord_learner_turtle[1] -= self.move_step

            # FIXME codice quasi esattamente duplicato da keep_in_screen()
            # Per evitare che lo Slime learner esca dallo schermo
            if self.cord_learner_turtle[0] > self.width - 10:
                self.cord_learner_turtle[0] = self.width - 15
            elif self.cord_learner_turtle[0] < 10:
                self.cord_learner_turtle[0] = 15
            if self.cord_learner_turtle[1] > self.height - 10:
                self.cord_learner_turtle[1] = self.height - 15
            elif self.cord_learner_turtle[1] < 10:
                self.cord_learner_turtle[1] = 15
        elif action == 1:  # DROP CHEMICALS
            for x in range(self.limit[0], self.limit[2]):
                for y in range(self.limit[1], self.limit[3]):
                    self.chemicals_level[str(x) + str(y)] += 2  # QUESTION dunque viene depositata la stessa quantità di feromone nell'area tra i limiti fissati?
        elif action == 2:  # CHASE MAX CHEMICAL
            # FIXME codice quasi esattamente duplicato da find_max_lv()
            for x in range(self.limit[0], self.limit[2]):
                for y in range(self.limit[1], self.limit[3]):
                    if self.chemicals_level[str(x) + str(y)] > self.max:  # CERCO IL MAX VALORE DI FEROMONE NELLE VICINANZE E PRENDO LE SUE COORDINATE
                        self.max = self.chemicals_level[str(x) + str(y)]
                        self.cord_max.append(x)
                        self.cord_max.append(y)
            if self.max > self.sniff_threshold:
                # FIXME codice quasi esattamente duplicato da follow_pheromone()
                if self.cord_max[0] > self.cord_learner_turtle[0] and self.cord_max[1] > self.cord_learner_turtle[1]:
                    self.cord_learner_turtle[0] += self.move_step
                    self.cord_learner_turtle[1] += self.move_step
                elif self.cord_max[0] < self.cord_learner_turtle[0] and self.cord_max[1] < self.cord_learner_turtle[1]:
                    self.cord_learner_turtle[0] -= self.move_step
                    self.cord_learner_turtle[1] -= self.move_step
                elif self.cord_max[0] > self.cord_learner_turtle[0] and self.cord_max[1] < self.cord_learner_turtle[1]:
                    self.cord_learner_turtle[0] += self.move_step
                    self.cord_learner_turtle[1] -= self.move_step
                elif self.cord_max[0] < self.cord_learner_turtle[0] and self.cord_max[1] > self.cord_learner_turtle[1]:
                    self.cord_learner_turtle[0] -= self.move_step
                    self.cord_learner_turtle[1] += self.move_step
                elif self.cord_max[0] < self.cord_learner_turtle[0] and self.cord_max[1] == self.cord_learner_turtle[1]:
                    self.cord_learner_turtle[0] -= self.move_step
                elif self.cord_max[0] > self.cord_learner_turtle[0] and self.cord_max[1] == self.cord_learner_turtle[1]:
                    self.cord_learner_turtle[0] += self.move_step
                elif self.cord_max[0] == self.cord_learner_turtle[0] and self.cord_max[1] < self.cord_learner_turtle[1]:
                    self.cord_learner_turtle[1] -= self.move_step
                elif self.cord_max[0] == self.cord_learner_turtle[0] and self.cord_max[1] > self.cord_learner_turtle[1]:
                    self.cord_learner_turtle[1] += self.move_step
                else:
                    pass
            else:
                pass

        cur_reward = Slime.rewardfunc7(self)  # <--reward function in uso

        # EVAPORATE CHEMICAL
        self._evaporate()

        self.observation = Slime._get_obs(self)

        return self.observation, cur_reward, False, {}

    def drop_chemical(self):
        """
        Action 1: drop chemical in patch where turtle is
        :return:
        """
        for x in range(self.bonds[0], self.bonds[2]):
            for y in range(self.bonds[1], self.bonds[3]):
                self.chemicals_level[str(x) + str(
                    y)] += 2  # TODO rendere parametrica la quantità di feromone, come 'chemical-drop' in netlogo

    def _evaporate(self):
        """
        evaporate pheromone
        :return:
        """
        for patch in self.chemicals_level:
            if self.chemicals_level[patch] != 0:
                self.chemicals_level[patch] -= 2  # TODO rendere parametrico come 'evaporation-rate' in netlogo

    def _keep_in_screen(self, turtle):
        """
        keep turtles within screen
        :param turtle:
        :return:
        """
        if self.cord_non_learner_turtle[turtle][0] > self.width - 10:
            self.cord_non_learner_turtle[turtle][0] = self.width - 15
        elif self.cord_non_learner_turtle[turtle][0] < 10:
            self.cord_non_learner_turtle[turtle][0] = 15
        if self.cord_non_learner_turtle[turtle][1] > self.height - 10:
            self.cord_non_learner_turtle[turtle][1] = self.height - 15
        elif self.cord_non_learner_turtle[turtle][1] < 10:
            self.cord_non_learner_turtle[turtle][1] = 15

    def rng_walk(self, turtle):
        """
        Action 0: move in random direction
        :param turtle:
        :return:
        """
        act = np.random.randint(4)
        if act == 0:
            self.cord_non_learner_turtle[turtle][0] += self.move_step
            self.cord_non_learner_turtle[turtle][1] += self.move_step
        elif act == 1:
            self.cord_non_learner_turtle[turtle][0] -= self.move_step
            self.cord_non_learner_turtle[turtle][1] -= self.move_step
        elif act == 2:
            self.cord_non_learner_turtle[turtle][0] -= self.move_step
            self.cord_non_learner_turtle[turtle][1] += self.move_step
        else:
            self.cord_non_learner_turtle[turtle][0] += self.move_step
            self.cord_non_learner_turtle[turtle][1] -= self.move_step

    def follow_pheromone(self, turtle):
        """
        Action 2: move turtle towards greatest pheromone found by _find_max_lv()
        :param turtle: the turtle to move
        :return: the new turtle x,y
        """
        if self.cord_max_lv[0] > self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] > \
                self.cord_non_learner_turtle[turtle][1]:
            # allora il punto si trova in alto a dx
            self.cord_non_learner_turtle[turtle][0] += self.move_step
            self.cord_non_learner_turtle[turtle][1] += self.move_step
        elif self.cord_max_lv[0] < self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] < \
                self.cord_non_learner_turtle[turtle][1]:
            # allora il punto si trova in basso a sx
            self.cord_non_learner_turtle[turtle][0] -= self.move_step
            self.cord_non_learner_turtle[turtle][1] -= self.move_step
        elif self.cord_max_lv[0] > self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] < \
                self.cord_non_learner_turtle[turtle][1]:
            # allora il punto si trova in basso a dx
            self.cord_non_learner_turtle[turtle][0] += self.move_step
            self.cord_non_learner_turtle[turtle][1] -= self.move_step
        elif self.cord_max_lv[0] < self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] > \
                self.cord_non_learner_turtle[turtle][1]:
            # allora il punto si trova in alto a sx
            self.cord_non_learner_turtle[turtle][0] -= self.move_step
            self.cord_non_learner_turtle[turtle][1] += self.move_step
        elif self.cord_max_lv[0] == self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] < \
                self.cord_non_learner_turtle[turtle][1]:
            # allora il punto si trova in basso sulla mia colonna
            self.cord_non_learner_turtle[turtle][1] -= self.move_step
        elif self.cord_max_lv[0] == self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] > \
                self.cord_non_learner_turtle[turtle][1]:
            # allora il punto si trova in alto sulla mia colonna
            self.cord_non_learner_turtle[turtle][1] += self.move_step
        elif self.cord_max_lv[0] > self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] == \
                self.cord_non_learner_turtle[turtle][1]:
            # allora il punto si trova alla mia dx
            self.cord_non_learner_turtle[turtle][0] += self.move_step
        elif self.cord_max_lv[0] < self.cord_non_learner_turtle[turtle][0] and self.cord_max_lv[1] == \
                self.cord_non_learner_turtle[turtle][1]:
            # allora il punto si trova alla mia sx
            self.cord_non_learner_turtle[turtle][0] -= self.move_step
        else:
            pass  # allora il punto è dove mi trovo quindi stò fermo

    def _find_max_lv(self, turtle):
        """
        find patch where chemical pheromone level is max within radius
        :param turtle: the sniffing turtle
        :return: x,y of patch with max chemical within radius
        """
        # DOC raggio entro cui cercare feromone
        self.bonds.append(self.cord_non_learner_turtle[turtle][0] - 3)
        self.bonds.append(self.cord_non_learner_turtle[turtle][1] - 3)
        self.bonds.append(self.cord_non_learner_turtle[turtle][0] + 4)
        self.bonds.append(self.cord_non_learner_turtle[turtle][1] + 4)
        for i in range(len(self.bonds)):
            if self.bonds[i] < 0:
                self.bonds[i] = 0
            elif self.bonds[i] > self.width:
                self.bonds[i] = self.width
        for x in range(self.bonds[0], self.bonds[2]):
            for y in range(self.bonds[1], self.bonds[3]):  # SCORRO LE "PATCH" NELLE VICINANE CON UN r = 3
                if self.chemicals_level[str(x) + str(
                        y)] > self.max_lv:  # CERCO IL MAX VALORE DI FEROMONE NELLE VICINANZE E PRENDO LE SUE COORDINATE
                    self.max_lv = self.chemicals_level[str(x) + str(y)]
                    # self.cord_max_lv.clear()
                    self.cord_max_lv = []
                    self.cord_max_lv.append(x)
                    self.cord_max_lv.append(y)

    def _get_obs(self):
        # controllo la presenza di feromone o meno nella patch, da spostare QUESTION perchè "da spostare"?
        self._check_chemical()

        # controllo if in cluster
        self._count_cluster()

        if self.count_turtle >= self.cluster_threshold:
            self.observation[0] = True
        else:
            self.observation[0] = False

        return self.observation

    def _count_cluster(self):
        self.count_turtle = 1
        self.check_cord = []
        for x in range(self.cord_learner_turtle[0] - self.cluster_radius//2, self.cord_learner_turtle[0] + self.cluster_radius//2):
            for y in range(self.cord_learner_turtle[1] - self.cluster_radius//2, self.cord_learner_turtle[1] + self.cluster_radius//2):
                self.check_cord.append([x, y])
        for pair in self.cord_non_learner_turtle.values():
            if pair in self.check_cord:
                self.count_turtle += 1

    def _check_chemical(self):
        if self.chemicals_level[str(self.cord_learner_turtle[0]) + str(self.cord_learner_turtle[1])] != 0:
            self.observation[1] = True
        else:
            self.observation[1] = False

    def rewardfunc1(self):
        self.count_turtle = 1
        self.check_cord = []
        for x in range(self.cord_learner_turtle[0] - 9, self.cord_learner_turtle[0] + 10):
            for y in range(self.cord_learner_turtle[1] - 9, self.cord_learner_turtle[1] + 10):
                self.check_cord.append([x, y])
        for pair in self.cord_non_learner_turtle.values():
            if pair in self.check_cord:
                self.count_turtle += 1
        if self.count_turtle >= self.cluster_threshold:
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
        if self.count_turtle >= self.cluster_threshold:
            self.count_ticks_cluster += 1
        else:
            self.count_ticks_cluster = 0
        if self.count_ticks_cluster > 1:
            self.reward_list.append(self.count_ticks_cluster)  # monotonic reward based on ticks in cluster

        return self.count_ticks_cluster

    def rewardfunc7(self):
        """
        reward is (positve) proportional to cluster size (quadratic) and (negative) proportional to time spent outside clusters
        :return:
        """
        self._count_cluster()
        if self.count_turtle >= self.cluster_threshold:
            self.count_ticks_cluster += 1

        # calcolo la reward
        cur_reward = ((self.count_turtle ^ 2) / self.cluster_threshold) * self.reward \
                     + \
                     (((ticks_per_episode - self.count_ticks_cluster) / ticks_per_episode) * self.penalty)

        self.reward_list.append(cur_reward)
        return cur_reward

    def reset(self):
        self.reward_list = []
        self.observation = [False, False]
        self.count_ticks_cluster = 0

        # create learner turtle
        self.cord_learner_turtle = []
        self.cord_learner_turtle.append(np.random.randint(10, self.width - 10))
        self.cord_learner_turtle.append(np.random.randint(10, self.height - 10))

        # create NON learner turtle
        self.cord_non_learner_turtle = {}
        for p in range(self.population):
            self.l = []
            self.l.append(np.random.randint(10, self.width - 10))
            self.l.append(np.random.randint(10, self.height - 10))
            self.cord_non_learner_turtle[str(p)] = self.l

        # patches-own [chemical] - amount of pheromone in the patch
        self.chemicals_level = {}
        for x in range(self.width + 1):
            for y in range(self.height + 1):
                self.chemicals_level[str(x) + str(y)] = 0
        return self.observation, 0, False, {}  # NB check if 0 makes sense

    def render(self, **kwargs):
        if self.first_gui:
            self.first_gui = False
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("SLIME")
            self.clock = pygame.time.Clock()
        self.screen.fill((0, 0, 0))
        #self.clock = pygame.time.Clock()
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
episodes = 100
ticks_per_episode = 500
# consigliabile almeno 500 tick_per_episode, altrimenti difficile vedere fenomeni di aggregazione

env = Slime(sniff_threshold=12, step=5, cluster_threshold=5, population=100, grid_size=500, rew=100, penalty=-1)
for ep in range(1, episodes+1):
    env.reset()
    print(f"EPISODE: {ep}")
    for tick in range(ticks_per_episode):
        observation, reward, done, info = env.step(env.action_space.sample())
        # if tick % 2 == 0:
        print(observation, reward)
        env.render()
env.close()
