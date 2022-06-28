import json
from typing import Optional

import gym
import pygame
from gym import spaces

import numpy as np
import random

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
RED = (190, 0, 0)


class BooleanSpace(gym.Space):  # TODO improve implementation: should be a N-dimensional space of N boolean values
    def __init__(self, size=None):
        """
        A space of boolean values
        :param size: how many boolean values the space is made of
        """
        assert isinstance(size, int) and size > 0
        self.size = size
        self.values = [False for _ in range(self.size)]
        gym.Space.__init__(self, (), bool)

    def contains(self, x):
        return x in self.values

    def sample(self):
        return [random.choice([True, False]) for _ in range(self.size)]
        # return self.values

    def observe(self):
        """
        Get the current observation
        :return: the current observation
        """
        return self.values

    def change(self, p, value):
        """
        Set a specific boolean value for the current observation
        :param p: which boolean values to change (position index)
        :param value: the boolean value to set
        :return: None
        """
        self.values[p] = value

    def change_all(self, values):
        """
        Set all the boolean values for the current observation
        :param values: the boolean values to set
        :return: None
        """
        self.values = values


class Slime(gym.Env):
    metadata = {"render_modes": "human", "render_fps": 30}

    def __init__(self,
                 render_mode: Optional[str] = None,
                 **kwargs):
        """
        :param population:          Controls the number of non-learning slimes (= green turtles)
        :param sniff_threshold:     Controls how sensitive slimes are to pheromone (higher values make slimes less
                                    sensitive to pheromone)—unclear effect on learning, could be negligible
        :param diffuse_area         Controls the diffusion radius
        :param smell_area:          Controls the radius of the square area sorrounding the turtle whithin which it smells pheromone
        :param lay_area:            Controls the radius of the square area sorrounding the turtle where pheromone is laid
        :param lay_amount:          Controls how much pheromone is laid
        :param evaporation:         Controls how much pheromone evaporates at each step
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
        :param episode_ticks:       Number of ticks for episode termination
        :param step:                How many pixels do turtle move at each movement step
        :param render_mode:
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.population = kwargs['population']
        self.sniff_threshold = kwargs['sniff_threshold']
        self.diffuse_area = kwargs['diffuse_area']
        self.smell_area = kwargs['smell_area']
        self.lay_area = kwargs['lay_area']
        self.lay_amount = kwargs['lay_amount']
        self.evaporation = kwargs['evaporation']
        self.cluster_threshold = kwargs['cluster_threshold']
        self.cluster_radius = kwargs['cluster_radius']
        self.reward = kwargs['rew']
        self.penalty = kwargs['penalty']
        self.episode_ticks = kwargs['episode_ticks']

        self.W = kwargs['W']
        self.H = kwargs['H']
        self.patch_size = kwargs['PATCH_SIZE']
        self.turtle_size = kwargs['TURTLE_SIZE']
        self.show_patches = kwargs['SHOW_PATCHES']

        self.coords = []
        self.offset = self.patch_size // 2
        self.W_pixels = self.W * self.patch_size
        self.H_pixels = self.H * self.patch_size
        for x in range(self.offset, (self.W_pixels - self.offset) + 1, self.patch_size):
            for y in range(self.offset, (self.H_pixels - self.offset) + 1, self.patch_size):
                self.coords.append((x, y))  # "centre" of the patch or turtle (also ID of the patch)

        self.screen = pygame.display.set_mode((self.W_pixels, self.H_pixels))
        self.clock = pygame.time.Clock()

        self.rewards = []
        self.cluster_ticks = 0  # conta i tick che la turtle passa in un cluster

        self.first_gui = True

        # create learner turtle
        self.learner = {"pos": self.coords[np.random.randint(len(self.coords))]}
        # create NON learner turtles
        self.turtles = {i: {"pos": self.coords[np.random.randint(len(self.coords))]} for i in range(self.population)}

        # patches-own [chemical] - amount of pheromone in each patch
        self.patches = {self.coords[i]: {"id": i, 'chemical': 0.0} for i in range(len(self.coords))}
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed smell area for each patch, including itself
        self.smell_patches = {}
        self._find_neighbours(self.smell_patches, self.smell_area)
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed lay area for each patch, including itself
        self.lay_patches = {}
        self._find_neighbours(self.lay_patches, self.lay_area)
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed diffusion area for each patch, including itself
        self.diffuse_patches = {}
        self._find_neighbours(self.diffuse_patches, self.diffuse_area)

        self.action_space = spaces.Discrete(3)  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone TODO as dict
        self.observation_space = BooleanSpace(size=2)  # DOC [0] = whether the turtle is in a cluster
        # DOC [1] = whether there is chemical in turtle patch

    def _find_neighbours(self, neighbours, area):
        """

        :param neighbours:
        :param area:
        :return: None (1st argument modified as side effect)
        """
        for p in self.patches:
            neighbours[p] = []
            for x in range(p[0], p[0] + (area * self.patch_size) + 1, self.patch_size):
                for y in range(p[1], p[1] + (area * self.patch_size) + 1, self.patch_size):
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] - (area * self.patch_size) - 1, -self.patch_size):
                for y in range(p[1], p[1] - (area * self.patch_size) - 1, -self.patch_size):
                    neighbours[p].append((x, y))
            neighbours[p] = list(set(neighbours[p]))

    def step(self, action: int):
        # DOC action: 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
        # non learners act
        for turtle in self.turtles:
            max_pheromone, max_coords = self._find_max_pheromone(self.turtles[turtle]['pos'], self.smell_area)

            if max_pheromone >= self.sniff_threshold:
                self.follow_pheromone(max_coords, self.turtles[turtle])
            else:
                self.walk(self.turtles[turtle])
            # self.walk(self.non_learner_pos[turtle])

            self.lay_pheromone(self.turtles[turtle]['pos'], self.lay_amount)

        # learner acts
        if action == 0:  # DOC walk
            self.walk(self.learner)
        elif action == 1:  # DOC lay_pheromone
            self.lay_pheromone(self.learner['pos'], self.lay_amount)
        elif action == 2:  # DOC follow_pheromone
            max_pheromone, max_coords = self._find_max_pheromone(self.learner['pos'], self.smell_area)
            if max_pheromone >= self.sniff_threshold:
                self.follow_pheromone(max_coords, self.learner)
            else:
                self.walk(self.learner)

        cur_reward = self.rewardfunc7()
        self.observation_space.change_all([self._check_cluster(), self._check_chemical()])

        self._diffuse()
        self._evaporate()

        return self.observation_space.observe(), cur_reward, False, {}

    def lay_pheromone(self, pos, amount):
        """
        Lay 'amount' pheromone in square 'area' centred in 'pos'
        :param pos: the x,y position taken as centre of pheromone deposit area
        :param amount: the amount of pheromone to deposit
        :return: None (environment properties are changed as side effect)
        """
        for p in self.lay_patches[pos]:
            self.patches[p]['chemical'] += amount

    def _diffuse(self):
        """

        :return:
        """
        for patch in self.patches:
            if self.patches[patch]['chemical'] > 0:
                n_size = len(self.diffuse_patches[patch])
                for n in self.diffuse_patches[patch]:
                    self.patches[n] += self.patches[patch] / (n_size + 1)
                self.patches[patch] = 1 / (n_size + 1)

    def _evaporate(self):
        """

        :return:
        """
        for patch in self.patches:
            if self.patches[patch] > 0:
                self.patches[patch] *= self.evaporation

    def walk(self, turtle):
        """
        Action 0: move in random direction (8 sorrounding cells
        :param turtle: the turtle to move
        :return: None (pos is updated after movement as side-effect)
        """
        choice = [self.patch_size, -self.patch_size, 0]
        x, y = turtle['pos']
        x2, y2 = x + np.random.choice(choice), y + np.random.choice(choice)
        if x2 < 0:
            x2 = self.W_pixels - self.offset
        if x2 > self.W_pixels:
            x2 = 0 + self.offset
        if y2 < 0:
            y2 = self.H_pixels - self.offset
        if y2 > self.H_pixels:
            y2 = 0 + self.offset
        turtle['pos'] = (x2, y2)

    def follow_pheromone(self, ph_coords, turtle):
        """
        Action 2: move turtle towards greatest pheromone found
        :param ph_coords: the position where max pheromone has been sensed
        :param turtle: the turtle looking for pheromone
        :return: None (pos is updated after movement as side-effect)
        """
        x, y = turtle['pos']
        if ph_coords[0] > x and ph_coords[1] > y:  # allora il punto si trova in alto a dx
            x += self.patch_size
            y += self.patch_size
        elif ph_coords[0] < x and ph_coords[1] < y:  # allora il punto si trova in basso a sx
            x -= self.patch_size
            y -= self.patch_size
        elif ph_coords[0] > x and ph_coords[1] < y:  # allora il punto si trova in basso a dx
            x += self.patch_size
            y -= self.patch_size
        elif ph_coords[0] < x and ph_coords[1] > y:  # allora il punto si trova in alto a sx
            x -= self.patch_size
            y += self.patch_size
        elif ph_coords[0] == x and ph_coords[1] < y:  # allora il punto si trova in basso sulla mia colonna
            y -= self.patch_size
        elif ph_coords[0] == x and ph_coords[1] > y:  # allora il punto si trova in alto sulla mia colonna
            y += self.patch_size
        elif ph_coords[0] > x and ph_coords[1] == y:  # allora il punto si trova alla mia dx
            x += self.patch_size
        elif ph_coords[0] < x and ph_coords[1] == y:  # allora il punto si trova alla mia sx
            x -= self.patch_size
        else:  # DOC il punto è la mia stessa patch
            pass
        turtle['pos'] = (x, y)

    def _find_max_pheromone(self, pos, area):
        """
        Find where the maximum pheromone level is within square 'area' centred in 'pos'
        :param pos: the x,y position of the turtle looking for pheromone
        :param area: the square area where to look within
        :return: the maximum pheromone level found and its x,y position
        """
        bounds = self._get_bounds(area, pos)

        max_ph = -1
        max_pos = []
        for x in range(bounds[0], bounds[2]):
            for y in range(bounds[1], bounds[3]):
                if self.chemical_pos[(x, y)] > max_ph:
                    max_ph = self.chemical_pos[(x, y)]
                    max_pos = [x, y]

        return max_ph, max_pos

    def _check_cluster(self):
        """
        Checks whether the learner turtle is within a cluster, given 'cluster_radius' and 'cluster_threshold'
        :return: a boolean
        """
        cluster = 1
        area = []
        for x in range(self.learner_pos[0] - self.cluster_radius // 2, self.learner_pos[0] + self.cluster_radius // 2):
            for y in range(self.learner_pos[1] - self.cluster_radius // 2,
                           self.learner_pos[1] + self.cluster_radius // 2):
                area.append([x, y])
        for pair in self.non_learner_pos.values():
            if pair in area:
                cluster += 1
        return cluster >= self.cluster_threshold

    def _check_chemical(self):
        """
        Checks whether there is pheromone on the patch where the learner turtle is
        :return: a boolean
        """
        return self.chemical_pos[(self.learner_pos[0], self.learner_pos[1])] > 0

    def rewardfunc7(self):
        """
        Reward is (positve) proportional to cluster size (quadratic) and (negative) proportional to time spent outside
        clusters
        :return: the reward
        """
        cluster = self._check_cluster()
        if cluster >= self.cluster_threshold:
            self.cluster_ticks += 1

        cur_reward = ((cluster ^ 2) / self.cluster_threshold) * self.reward + (
                ((self.episode_ticks - self.cluster_ticks) / self.episode_ticks) * self.penalty)

        self.rewards.append(cur_reward)
        return cur_reward

    def reset(self):
        # empty stuff
        self.rewards = []
        self.observation_space.change_all([False, False])
        self.cluster_ticks = 0

        # re-position learner turtle
        self.learner['pos'] = self.coords[np.random.randint(len(self.coords))]
        # re-position NON learner turtles
        for t in self.turtles:
            self.turtles[id]['pos'] = self.coords[np.random.randint(len(self.coords))]
        # patches-own [chemical] - amount of pheromone in the patch
        for p in self.patches:
            self.patches[p]['chemical'] = 0.0

        return self.observation_space.observe(), 0, False, {}  # TODO check if 0 makes sense

    def render(self, **kwargs):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # chiusura finestra -> termina il programma
                pygame.quit()
        if self.first_gui:
            self.first_gui = False
            pygame.init()
            pygame.display.set_caption("SLIME")

        self.screen.fill(BLACK)
        self.clock.tick(self.metadata["render_fps"])

        # Disegno LA turtle learner!
        pygame.draw.circle(self.screen, RED, (self.learner_pos[0], self.learner_pos[1]), 3)

        for turtle in self.non_learner_pos.values():
            pygame.draw.circle(self.screen, BLUE, (turtle[0], turtle[1]), 3)
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


#   MAIN
PARAMS_FILE = "SlimeEnvV2-params.json"
EPISODES = 10
LOG_EVERY = 100

with open(PARAMS_FILE) as f:
    params = json.load(f)
env = Slime(render_mode="human", **params)
for ep in range(1, EPISODES + 1):
    env.reset()
    print(f"-------------------------------------------\nEPISODE: {ep}\n-------------------------------------------")
    for tick in range(params['episode_ticks']):
        observation, reward, done, info = env.step(env.action_space.sample())
        if tick % LOG_EVERY == 0:
            print(f"{tick}: {observation}, {reward}")
        env.render()
env.close()
