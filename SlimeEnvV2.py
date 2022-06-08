from typing import Optional

import gym
import pygame
from gym import spaces

import numpy as np
import random


class BooleanSpace(gym.Space):  # TODO improve implementation: should be a N-dimensional space of N boolean values
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
        #return self.values

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

    def change(self, values):
        """
        Set all the boolean values for the current observation
        :param values: the boolean values to set
        :return: None
        """
        self.values = values


class Slime(gym.Env):
    metadata = {"render_modes": "human", "render_fps": 30}

    def __init__(self,
                 population=100,
                 sniff_threshold=1,
                 smell_area=2,
                 lay_area=1,
                 lay_amount=3,
                 evaporation=0.9,
                 cluster_threshold=20,
                 cluster_radius=10,
                 rew=100,
                 penalty=-1,
                 episode_ticks=500,
                 step=1,
                 grid_size=250,
                 render_mode: Optional[str] = None):
        """
        :param population:          Controls the number of non-learning slimes (= green turtles)
        :param sniff_threshold:     Controls how sensitive slimes are to pheromone (higher values make slimes less
                                    sensitive to pheromone)—unclear effect on learning, could be negligible
        :param smell_area:          Controls the square area sorrounding the turtle whithin which it smells pheromone
        :param lay_area:            Controls the square area sorrounding the turtle where pheromone is laid
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
        :param grid_size:           Simulation area is always a square
        :param render_mode:
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.population = population
        self.sniff_threshold = sniff_threshold
        self.smell_area = smell_area
        self.lay_area = lay_area
        self.lay_amount = lay_amount
        self.evaporation = evaporation
        self.cluster_threshold = cluster_threshold
        self.cluster_radius = cluster_radius
        self.reward = rew
        self.penalty = penalty
        self.episode_ticks = episode_ticks

        self.move_step = step
        self.width = grid_size
        self.height = grid_size

        self.rewards = []
        self.cluster_ticks = 0    # conta i tick che la turtle passa in un cluster

        self.first_gui = True

        # create learner turtle
        self.learner_pos = [np.random.randint(10, self.width-10) for _ in range(2)]  # QUESTION +10 / -10 è per non mettere turtles troppo vicine al bordo?
        # create NON learner turtles
        self.non_learner_pos = {}
        for p in range(self.population):
            self.non_learner_pos[str(p)] = [np.random.randint(10, self.width-10) for _ in range(2)]

        # patches-own [chemical] - amount of pheromone in each patch
        self.chemical_pos = {}
        for x in range(self.width + 1):
            for y in range(self.height + 1):
                self.chemical_pos[(x, y)] = 0.0

        self.action_space = spaces.Discrete(3)          # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone TODO as dict
        self.observation_space = BooleanSpace(size=2)   # DOC [0] = whether the turtle is in a cluster
                                                        # DOC [1] = whether there is chemical in turtle patch
        self.observation = [False, False]   # FIXME di fatto non usi lo spazio in questo modo

    def step(self, action: int):
        # DOC action: 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
        # non learners act
        for turtle in self.non_learner_pos:
            max_pheromone, max_coords = self._find_max_pheromone(self.non_learner_pos[turtle], self.smell_area)

            if max_pheromone >= self.sniff_threshold:
                self.follow_pheromone(max_coords, self.non_learner_pos[turtle])
            else:
                self.walk(self.non_learner_pos[turtle])

            self.lay_pheromone(self.non_learner_pos[turtle], self.lay_area, self.lay_amount)
            self._wrap(self.non_learner_pos[turtle])

        # learner acts
        if action == 0:  # DOC walk
            self.walk(self.learner_pos)
            self._wrap(self.learner_pos)
        elif action == 1:  # DOC lay_pheromone
            self.lay_pheromone(self.learner_pos, self.lay_area, self.lay_amount)
        elif action == 2:  # DOC follow_pheromone
            max_pheromone, max_coords = self._find_max_pheromone(self.learner_pos, self.smell_area)
            if max_pheromone >= self.sniff_threshold:
                self.follow_pheromone(max_coords, self.learner_pos)
            else:
                pass  # TODO check
            self._wrap(self.learner_pos)

        cur_reward = self.rewardfunc7()
        self.observation = self._get_obs()

        self._diffuse()
        self._evaporate()

        return self.observation, cur_reward, False, {}

    def lay_pheromone(self, pos, area, amount):
        """
        Lay 'amount' pheromone in square 'area' centred in 'pos'
        :param pos: the x,y position taken as centre of pheromone deposit area
        :param area: the square area within which pheromone will be laid
        :param amount: the amount of pheromone to deposit
        :return: None (environment properties are changed as side effect)
        """
        bounds = self._get_bounds(area, pos)
        for x in range(bounds[0], bounds[2]):
            for y in range(bounds[1], bounds[3]):
                self.chemical_pos[(x, y)] += amount

    def _get_bounds(self, area, pos):
        """
        Computes x,y bounds of a square area big as 'area' centred in 'pos'
        :param area: the size of the square area
        :param pos: the x,y centre of the area
        :return: the x,y of the bounds of the area (list with 4 elements [minx, miny, maxx, maxy])
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
        return bounds

    def _diffuse(self):
        for patch in self.chemical_pos:
            if self.chemical_pos[patch] > 0:
                neighbours = self.adj_finder(patch)
                for pos in neighbours:
                    self.chemical_pos[pos] += self.chemical_pos[patch] * 1 / (len(neighbours) + 1)
                self.chemical_pos[patch] = 1 / (len(neighbours) + 1)

    def adj_finder(self, position, distance=1):
        """
        Generate adjacent cells within 'distance' to 'position' in every direction (diagonal included) that are legit,
        that is, that do not extend beyond grid size and that are not the reference 'position'
        :param position:
        :param distance:
        :return:
        """
        adj = []

        for dx in range(0-distance, 1+distance):
            for dy in range(0-distance, 1+distance):
                range_x = range(0, self.width+1)  # X bounds
                range_y = range(0, self.height+1)  # Y bounds

                (new_x, new_y) = (position[0] + dx, position[1] + dy)  # adjacent cell

                if (new_x in range_x) and (new_y in range_y) and (dx, dy) != (0, 0):
                    adj.append((new_x, new_y))

        return adj

    def _evaporate(self):
        """

        :return:
        """
        for patch in self.chemical_pos:
            if self.chemical_pos[patch] > 0:
                self.chemical_pos[patch] *= self.evaporation

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

    def walk(self, pos):
        """
        Action 0: move in random direction
        :param pos: the x,y position of the turtle to move
        :return: None (pos is updated after movement as side-effect)
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

    def follow_pheromone(self, ph_coords, pos):
        """
        Action 2: move turtle towards greatest pheromone found
        :param ph_coords: the position where max pheromone has been sensed
        :param pos: the x,y position of the turtle looking for pheromone
        :return: None (pos is updated after movement as side-effect)
        """
        if ph_coords[0] > pos[0] and ph_coords[1] > pos[1]:  # allora il punto si trova in alto a dx
            pos[0] += self.move_step
            pos[1] += self.move_step
        elif ph_coords[0] < pos[0] and ph_coords[1] < pos[1]:  # allora il punto si trova in basso a sx
            pos[0] -= self.move_step
            pos[1] -= self.move_step
        elif ph_coords[0] > pos[0] and ph_coords[1] < pos[1]:  # allora il punto si trova in basso a dx
            pos[0] += self.move_step
            pos[1] -= self.move_step
        elif ph_coords[0] < pos[0] and ph_coords[1] > pos[1]:  # allora il punto si trova in alto a sx
            pos[0] -= self.move_step
            pos[1] += self.move_step
        elif ph_coords[0] == pos[0] and ph_coords[1] < pos[1]:  # allora il punto si trova in basso sulla mia colonna
            pos[1] -= self.move_step
        elif ph_coords[0] == pos[0] and ph_coords[1] > pos[1]:  # allora il punto si trova in alto sulla mia colonna
            pos[1] += self.move_step
        elif ph_coords[0] > pos[0] and ph_coords[1] == pos[1]:  # allora il punto si trova alla mia dx
            pos[0] += self.move_step
        elif ph_coords[0] < pos[0] and ph_coords[1] == pos[1]:  # allora il punto si trova alla mia sx
            pos[0] -= self.move_step
        else:
            pass  # TODO check

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

    def _get_obs(self):
        """
        Get environment observations
        :return: the environment observations
        """
        self.observation[0] = self._check_cluster()
        # da spostare QUESTION perchè "da spostare"?
        self.observation[1] = self._check_chemical()

        return self.observation   # FIXME di fatto non usi lo spazio in questo modo

    def _check_cluster(self):
        """
        Checks whether the learner turtle is within a cluster, given 'cluster_radius' and 'cluster_threshold'
        :return: a boolean
        """
        cluster = 1
        area = []
        for x in range(self.learner_pos[0] - self.cluster_radius // 2, self.learner_pos[0] + self.cluster_radius // 2):
            for y in range(self.learner_pos[1] - self.cluster_radius // 2, self.learner_pos[1] + self.cluster_radius // 2):
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
            self.cluster_ticks += 1
            self.reward = 2
            self.rewards.append(2)   # se la mia turtle è in un cluster gli assegno una reward
        else:
            self.cluster_ticks = 0
            self.reward = -2
            self.rewards.append(-0.2)  # se la mia turtle NON è in un cluster gli assegno una penalty

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
            self.cluster_ticks += 1
        else:
            self.cluster_ticks = 0
        if self.cluster_ticks > 1:
            self.rewards.append(self.cluster_ticks)  # monotonic reward based on ticks in cluster

        return self.cluster_ticks

    def rewardfunc7(self):
        """
        Reward is (positve) proportional to cluster size (quadratic) and (negative) proportional to time spent outside clusters
        :return: the reward
        """
        cluster = self._check_cluster()
        if cluster >= self.cluster_threshold:
            self.cluster_ticks += 1

        cur_reward = ((cluster ^ 2) / self.cluster_threshold) * self.reward \
                     + \
                     (((self.episode_ticks - self.cluster_ticks) / self.episode_ticks) * self.penalty)

        self.rewards.append(cur_reward)
        return cur_reward

    def reset(self):
        # empty stuff
        self.rewards = []
        self.observation = [False, False]
        self.cluster_ticks = 0

        # re-position learner turtle
        self.learner_pos = [np.random.randint(10, self.width-10) for _ in range(2)]

        # re-position NON learner turtles
        self.non_learner_pos = {}
        for p in range(self.population):
            self.non_learner_pos[str(p)] = [np.random.randint(10, self.width - 10) for _ in range(2)]

        # patches-own [chemical] - amount of pheromone in the patch
        self.chemical_pos = {}
        for x in range(self.width + 1):
            for y in range(self.height + 1):
                self.chemical_pos[(x, y)] = 0
        return self.observation, 0, False, {}  # TODO check if 0 makes sense

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
EPISODES = 5
EPISODE_TICKS = 500
LOG_EVERY = 100

env = Slime(population=100,
            sniff_threshold=1,
            smell_area=2,
            lay_area=1,
            lay_amount=3,
            evaporation=0.9,
            cluster_threshold=20,
            cluster_radius=10,
            rew=100,
            penalty=-1,
            episode_ticks=EPISODE_TICKS,
            step=1,
            grid_size=250,
            render_mode="human")
for ep in range(1, EPISODES + 1):
    env.reset()
    print(f"-------------------------------------------\nEPISODE: {ep}\n-------------------------------------------")
    for tick in range(EPISODE_TICKS):
        observation, reward, done, info = env.step(env.action_space.sample())
        if tick % LOG_EVERY == 0:
            print(f"{tick}: {observation}, {reward}")
        env.render()
env.close()
