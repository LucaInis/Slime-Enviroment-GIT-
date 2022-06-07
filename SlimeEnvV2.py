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
        return [random.choice([True, False]) for _ in range(self.size)]
        #return self.values

    def observe(self):
        """

        :return:
        """
        return self.values

    def change(self, p, value):
        """

        :param p:
        :param value:
        :return:
        """
        self.values[p] = value

    def change(self, values):
        """

        :param values:
        :return:
        """
        self.values = values


class Slime(gym.Env):
    metadata = {"render_modes": "human", "render_fps": 30}

    def __init__(self,
                 population=650,
                 sniff_threshold=12,
                 smell_area=4,
                 lay_area=4,
                 lay_amount=2,
                 cluster_threshold=5,
                 cluster_radius=20,
                 rew=100,
                 penalty=-1,
                 render_mode: Optional[str] = None,
                 step=5,
                 grid_size=500):
        """

        :param population:          Controls the number of non-learning slimes (= green turtles)
        :param sniff_threshold:     Controls how sensitive slimes are to pheromone (higher values make slimes less
                                    sensitive to pheromone)—unclear effect on learning, could be negligible
        :param smell_area:          Controls the square area sorrounding the turtle whithin which it smells pheromone
        :param lay_area:            Controls the square area sorrounding the turtle where pheromone is laid
        :param lay_amount:          Controls how much pheromone is laid
        :param cluster_threshold:   Controls the minimum number of slimes needed to consider an aggregate within
                                    cluster-radius a cluster (the higher the more difficult to consider an aggregate a
                                    cluster)—the higher the more difficult to obtain a positive reward for being within
                                    a cluster for learning slimes
        :param cluster_radius:      Controls the range considered by slimes to count other slimes within a cluster (the
                                    higher the easier to form clusters, as turtles far apart are still counted together)
                                    —the higher the easier it is to obtain a positive reward for being within a cluster
                                    for learning slimes
        :param rew:                 Base reward for being in a cluster
        :param penalty:             Base penalty for not being in a cluster
        :param render_mode:
        :param step:                How many pixels do turtle move at each movement step
        :param grid_size:           Simulation area is always a square
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.population = population
        self.sniff_threshold = sniff_threshold
        self.smell_area = smell_area
        self.lay_area = lay_area
        self.lay_amount = lay_amount
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

        # DOC create learner turtle
        self.learner_pos = [np.random.randint(10, self.width-10) for _ in range(2)]  # QUESTION +10 / -10 è per non mettere turtles troppo vicine al bordo?
        # DOC create NON learner turtles
        self.non_learner_pos = {}
        for p in range(self.population):
            self.non_learner_pos[str(p)] = [np.random.randint(10, self.width-10) for _ in range(2)]

        # DOC patches-own [chemical] - amount of pheromone in each patch
        self.chemical_pos = {}
        for x in range(self.width + 1):
            for y in range(self.height + 1):
                self.chemical_pos[str(x) + str(y)] = 0

        self.action_space = spaces.Discrete(3)          # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone TODO as dict
        self.observation_space = BooleanSpace(size=2)   # DOC [0] = whether the turtle is in a cluster
                                                        # DOC [1] = whether there is chemical in turtle patch
        self.observation = [False, False]   # FIXME di fatto non usi lo spazio in questo modo

    def step(self, action: int):
        """

        :param action: 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
        :return:
        """
        # DOC non learners act
        for turtle in self.non_learner_pos:
            self.max_lv = 0  # TODO remove
            self.cord_max_lv = []  # TODO remove
            self.bonds = []  # TODO remove

            self._find_max_lv(turtle)  # TODO remove
            max_pheromone, max_coords = self._find_max_pheromone(self.non_learner_pos[turtle], self.smell_area)

            if max_pheromone > self.sniff_threshold:
                self.follow_pheromone(turtle)
            else:
                self.rng_walk(turtle)  # TODO remove
                self.walk(self.non_learner_pos[turtle])

            self.drop_chemical()  # TODO remove
            self.lay_pheromone(self.non_learner_pos[turtle], self.lay_area, self.lay_amount)
            self._keep_in_screen(turtle)

        # DOC learner act
        if action == 0:  # DOC random walk
            self.walk(self.learner_pos)
            self._wrap(self.learner_pos)
        elif action == 1:  # DROP CHEMICALS
            self.lay_pheromone(self.learner_pos, self.lay_area, self.lay_amount)
        elif action == 2:  # CHASE MAX CHEMICAL
            max_pheromone, max_coords = self._find_max_pheromone(self.learner_pos, self.smell_area)
            if max_pheromone > self.sniff_threshold:
                # FIXME codice quasi esattamente duplicato da follow_pheromone()
                if self.cord_max[0] > self.learner_pos[0] and self.cord_max[1] > self.learner_pos[1]:
                    self.learner_pos[0] += self.move_step
                    self.learner_pos[1] += self.move_step
                elif self.cord_max[0] < self.learner_pos[0] and self.cord_max[1] < self.learner_pos[1]:
                    self.learner_pos[0] -= self.move_step
                    self.learner_pos[1] -= self.move_step
                elif self.cord_max[0] > self.learner_pos[0] and self.cord_max[1] < self.learner_pos[1]:
                    self.learner_pos[0] += self.move_step
                    self.learner_pos[1] -= self.move_step
                elif self.cord_max[0] < self.learner_pos[0] and self.cord_max[1] > self.learner_pos[1]:
                    self.learner_pos[0] -= self.move_step
                    self.learner_pos[1] += self.move_step
                elif self.cord_max[0] < self.learner_pos[0] and self.cord_max[1] == self.learner_pos[1]:
                    self.learner_pos[0] -= self.move_step
                elif self.cord_max[0] > self.learner_pos[0] and self.cord_max[1] == self.learner_pos[1]:
                    self.learner_pos[0] += self.move_step
                elif self.cord_max[0] == self.learner_pos[0] and self.cord_max[1] < self.learner_pos[1]:
                    self.learner_pos[1] -= self.move_step
                elif self.cord_max[0] == self.learner_pos[0] and self.cord_max[1] > self.learner_pos[1]:
                    self.learner_pos[1] += self.move_step
                else:
                    pass
            else:
                pass

        cur_reward = Slime.rewardfunc7(self)  # <--reward function in uso

        # EVAPORATE CHEMICAL
        self._evaporate()

        self.observation = Slime._get_obs(self)

        return self.observation, cur_reward, False, {}

    def lay_pheromone(self, pos, area, amount):
        """

        :param pos:
        :param area:
        :param amount:
        :return:
        """
        bounds = [pos[0] - area // 2,  # DOC min x
                  pos[1] - area // 2,  # DOC min y
                  pos[0] + area // 2,  # DOC max x
                  pos[1] + area // 2]  # DOC max y
        for i in range(len(bounds)):
            if bounds[i] < 0:
                bounds[i] = 0
            elif bounds[i] > self.width:
                bounds[i] = self.width
        for x in range(self.bonds[0], self.bonds[2]):
            for y in range(self.bonds[1], self.bonds[3]):
                self.chemical_pos[str(x) + str(y)] += amount

    def drop_chemical(self):
        for x in range(self.bonds[0], self.bonds[2]):
            for y in range(self.bonds[1], self.bonds[3]):
                self.chemical_pos[str(x) + str(
                    y)] += 2  # TODO rendere parametrica la quantità di feromone, come 'chemical-drop' in netlogo

    def _evaporate(self):
        for patch in self.chemical_pos:
            if self.chemical_pos[patch] != 0:
                self.chemical_pos[patch] -= 2  # TODO rendere parametrico come 'evaporation-rate' in netlogo

    def _wrap(self, pos):
        """

        :param pos:
        :return:
        """
        if pos[0] > self.width - 10:  # QUESTION qual è il criterio per questi due numeri?
            pos[0] = self.width - 15
        elif pos[0] < 10:
            pos[0] = 15
        if pos[1] > self.height - 10:
            pos[1] = self.height - 15
        elif pos[1] < 10:
            pos[1] = 15

    def _keep_in_screen(self, turtle):
        if self.non_learner_pos[turtle][0] > self.width - 10:
            self.non_learner_pos[turtle][0] = self.width - 15
        elif self.non_learner_pos[turtle][0] < 10:
            self.non_learner_pos[turtle][0] = 15
        if self.non_learner_pos[turtle][1] > self.height - 10:
            self.non_learner_pos[turtle][1] = self.height - 15
        elif self.non_learner_pos[turtle][1] < 10:
            self.non_learner_pos[turtle][1] = 15

    def walk(self, pos):
        """
        Action 0: move in random direction
        :param pos: the x,y position of the turtle looking for pheromone
        :return: None (pos is updated after movement as side-effec)
        """
        act = np.random.randint(4)
        if act == 0:
            pos[0] += self.move_step
            pos[1] += self.move_step
        elif act == 1:
            pos[0] -= self.move_step
            pos[1] -= self.move_step
        elif act == 2:
            pos[0] -= self.move_step
            pos[1] += self.move_step
        else:
            pos[0] += self.move_step
            pos[1] -= self.move_step

    def rng_walk(self, turtle):
        act = np.random.randint(4)
        if act == 0:
            self.non_learner_pos[turtle][0] += self.move_step
            self.non_learner_pos[turtle][1] += self.move_step
        elif act == 1:
            self.non_learner_pos[turtle][0] -= self.move_step
            self.non_learner_pos[turtle][1] -= self.move_step
        elif act == 2:
            self.non_learner_pos[turtle][0] -= self.move_step
            self.non_learner_pos[turtle][1] += self.move_step
        else:
            self.non_learner_pos[turtle][0] += self.move_step
            self.non_learner_pos[turtle][1] -= self.move_step

    def follow_pheromone(self, turtle):
        """
        Action 2: move turtle towards greatest pheromone found by _find_max_lv()
        :param turtle: the turtle to move
        :return: the new turtle x,y
        """
        if self.cord_max_lv[0] > self.non_learner_pos[turtle][0] and self.cord_max_lv[1] > \
                self.non_learner_pos[turtle][1]:
            # allora il punto si trova in alto a dx
            self.non_learner_pos[turtle][0] += self.move_step
            self.non_learner_pos[turtle][1] += self.move_step
        elif self.cord_max_lv[0] < self.non_learner_pos[turtle][0] and self.cord_max_lv[1] < \
                self.non_learner_pos[turtle][1]:
            # allora il punto si trova in basso a sx
            self.non_learner_pos[turtle][0] -= self.move_step
            self.non_learner_pos[turtle][1] -= self.move_step
        elif self.cord_max_lv[0] > self.non_learner_pos[turtle][0] and self.cord_max_lv[1] < \
                self.non_learner_pos[turtle][1]:
            # allora il punto si trova in basso a dx
            self.non_learner_pos[turtle][0] += self.move_step
            self.non_learner_pos[turtle][1] -= self.move_step
        elif self.cord_max_lv[0] < self.non_learner_pos[turtle][0] and self.cord_max_lv[1] > \
                self.non_learner_pos[turtle][1]:
            # allora il punto si trova in alto a sx
            self.non_learner_pos[turtle][0] -= self.move_step
            self.non_learner_pos[turtle][1] += self.move_step
        elif self.cord_max_lv[0] == self.non_learner_pos[turtle][0] and self.cord_max_lv[1] < \
                self.non_learner_pos[turtle][1]:
            # allora il punto si trova in basso sulla mia colonna
            self.non_learner_pos[turtle][1] -= self.move_step
        elif self.cord_max_lv[0] == self.non_learner_pos[turtle][0] and self.cord_max_lv[1] > \
                self.non_learner_pos[turtle][1]:
            # allora il punto si trova in alto sulla mia colonna
            self.non_learner_pos[turtle][1] += self.move_step
        elif self.cord_max_lv[0] > self.non_learner_pos[turtle][0] and self.cord_max_lv[1] == \
                self.non_learner_pos[turtle][1]:
            # allora il punto si trova alla mia dx
            self.non_learner_pos[turtle][0] += self.move_step
        elif self.cord_max_lv[0] < self.non_learner_pos[turtle][0] and self.cord_max_lv[1] == \
                self.non_learner_pos[turtle][1]:
            # allora il punto si trova alla mia sx
            self.non_learner_pos[turtle][0] -= self.move_step
        else:
            pass  # allora il punto è dove mi trovo quindi stò fermo

    def _find_max_pheromone(self, pos, area):
        """

        :param pos: the x,y position of the turtle looking for pheromone
        :param area: the square area where to look within
        :return: the maximum pheromone level found and its x,y position
        """
        bounds = [pos[0] - area // 2,   # DOC min x
                  pos[1] - area // 2,   # DOC min y
                  pos[0] + area // 2,   # DOC max x
                  pos[1] + area // 2]   # DOC max y
        for i in range(len(bounds)):
            if bounds[i] < 0:
                bounds[i] = 0
            elif bounds[i] > self.width:
                bounds[i] = self.width

        max_ph = -1
        max_pos = []
        for x in range(self.bonds[0], self.bonds[2]):
            for y in range(self.bonds[1], self.bonds[3]):
                if self.chemical_pos[str(x) + str(y)] > max_ph:
                    max_ph = self.chemical_pos[str(x) + str(y)]
                    max_pos = [x, y]

        return max_ph, max_pos

    def _find_max_lv(self, turtle):
        # DOC raggio entro cui cercare feromone
        self.bonds.append(self.non_learner_pos[turtle][0] - 3)
        self.bonds.append(self.non_learner_pos[turtle][1] - 3)
        self.bonds.append(self.non_learner_pos[turtle][0] + 4)
        self.bonds.append(self.non_learner_pos[turtle][1] + 4)
        for i in range(len(self.bonds)):
            if self.bonds[i] < 0:
                self.bonds[i] = 0
            elif self.bonds[i] > self.width:
                self.bonds[i] = self.width
        for x in range(self.bonds[0], self.bonds[2]):
            for y in range(self.bonds[1], self.bonds[3]):  # SCORRO LE "PATCH" NELLE VICINANE CON UN r = 3
                if self.chemical_pos[str(x) + str(
                        y)] > self.max_lv:  # CERCO IL MAX VALORE DI FEROMONE NELLE VICINANZE E PRENDO LE SUE COORDINATE
                    self.max_lv = self.chemical_pos[str(x) + str(y)]
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
        for x in range(self.learner_pos[0] - self.cluster_radius // 2, self.learner_pos[0] + self.cluster_radius // 2):
            for y in range(self.learner_pos[1] - self.cluster_radius // 2, self.learner_pos[1] + self.cluster_radius // 2):
                self.check_cord.append([x, y])
        for pair in self.non_learner_pos.values():
            if pair in self.check_cord:
                self.count_turtle += 1

    def _check_chemical(self):
        if self.chemical_pos[str(self.learner_pos[0]) + str(self.learner_pos[1])] != 0:
            self.observation[1] = True
        else:
            self.observation[1] = False

    def rewardfunc1(self):
        self.count_turtle = 1
        self.check_cord = []
        for x in range(self.learner_pos[0] - 9, self.learner_pos[0] + 10):
            for y in range(self.learner_pos[1] - 9, self.learner_pos[1] + 10):
                self.check_cord.append([x, y])
        for pair in self.non_learner_pos.values():
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
        for pair in self.non_learner_pos.values():
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
        self.learner_pos = []
        self.learner_pos.append(np.random.randint(10, self.width - 10))
        self.learner_pos.append(np.random.randint(10, self.height - 10))

        # create NON learner turtle
        self.non_learner_pos = {}
        for p in range(self.population):
            self.l = []
            self.l.append(np.random.randint(10, self.width - 10))
            self.l.append(np.random.randint(10, self.height - 10))
            self.non_learner_pos[str(p)] = self.l

        # patches-own [chemical] - amount of pheromone in the patch
        self.chemical_pos = {}
        for x in range(self.width + 1):
            for y in range(self.height + 1):
                self.chemical_pos[str(x) + str(y)] = 0
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
        pygame.draw.circle(self.screen, (190, 0, 0), (self.learner_pos[0], self.learner_pos[1]), 3)

        for turtle in self.non_learner_pos.values():
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
