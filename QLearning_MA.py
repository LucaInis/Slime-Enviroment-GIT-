import json
from typing import Optional

import gym
import pygame
from gym import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

import numpy as np
import random


BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
RED = (190, 0, 0)
GREEN = (0, 190, 0)


class BooleanSpace(gym.Space):
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
        return [self.values for _ in range(self.size)]


class Slime(AECEnv):
    metadata = {"render_modes": "human"}

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
        :param render_mode:
        """
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.population = kwargs['population']
        self.learner_population = kwargs['learner_population']
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
        self.fps = kwargs['FPS']
        self.shade_strength = kwargs['SHADE_STRENGTH']
        self.show_chem_text = kwargs['SHOW_CHEM_TEXT']

        self.coords = []
        self.offset = self.patch_size // 2
        self.W_pixels = self.W * self.patch_size
        self.H_pixels = self.H * self.patch_size
        for x in range(self.offset, (self.W_pixels - self.offset) + 1, self.patch_size):
            for y in range(self.offset, (self.H_pixels - self.offset) + 1, self.patch_size):
                self.coords.append((x, y))  # "centre" of the patch or turtle (also ID of the patch)

        self.screen = pygame.display.set_mode((self.W_pixels, self.H_pixels))
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.cluster_font = pygame.font.SysFont("arial", self.patch_size // 2)
        self.chemical_font = pygame.font.SysFont("arial", self.patch_size // 3)

        self.agents = [i for i in range(self.learner_population)]
        # inizializzo il selettore
        self._agent_selector = agent_selector(self.agents)

        self.rewards = {i: [] for i in range(self.learner_population)}
        self.cluster_ticks = {i: 0 for i in range(self.learner_population)}

        self.first_gui = True

        n_coords = len(self.coords)
        # create learners turtle
        self.learners = {i: {"pos": self.coords[np.random.randint(n_coords)]} for i in range(self.learner_population)}
        # create NON learner turtles
        self.turtles = {i: {"pos": self.coords[np.random.randint(n_coords)]} for i in range(self.population)}

        # patches-own [chemical] - amount of pheromone in each patch
        self.patches = {self.coords[i]: {"id": i,
                                         'chemical': 0.0,
                                         'turtles': []} for i in range(n_coords)}
        for l in self.learners:
            self.patches[self.learners[l]['pos']]['turtles'].append(l)  # DOC id of learner turtle
        for t in self.turtles:
            self.patches[self.turtles[t]['pos']]['turtles'].append(t)
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed smell area for each patch, including itself
        self.smell_patches = {}
        self._find_neighbours(self.smell_patches, self.smell_area)
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed lay area for each patch, including itself
        self.lay_patches = {}
        self._find_neighbours(self.lay_patches, self.lay_area)
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed diffusion area for each patch, including itself
        self.diffuse_patches = {}
        self._find_neighbours(self.diffuse_patches, self.diffuse_area)
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed cluster-check for each patch, including itself
        self.cluster_patches = {}
        self._find_neighbours(self.cluster_patches, self.cluster_radius)

        self.action_spaces = {a: spaces.Discrete(3) for a in self.agents}  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
        self.observation_spaces = BooleanSpace(size=2)
        self.obs_dict = {a: self.observation_spaces.sample() for a in self.agents}  # DOC [0] = whether the turtle is in a cluster
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
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] - (area * self.patch_size) - 1, -self.patch_size):
                for y in range(p[1], p[1] - (area * self.patch_size) - 1, -self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] + (area * self.patch_size) + 1, self.patch_size):
                for y in range(p[1], p[1] - (area * self.patch_size) - 1, -self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] - (area * self.patch_size) - 1, -self.patch_size):
                for y in range(p[1], p[1] + (area * self.patch_size) + 1, self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            neighbours[p] = list(set(neighbours[p]))

    def _wrap(self, x, y):
        """
        Wrap x,y coordinates around the torus
        :param x:
        :param y:
        :return:
        """
        if x < 0:
            x = self.W_pixels - self.offset
        elif x > self.W_pixels:
            x = 0 + self.offset
        if y < 0:
            y = self.H_pixels - self.offset
        elif y > self.H_pixels:
            y = 0 + self.offset
        return x, y

    # learner acts
    def step(self, action: int):
        agent = self.agent_selection  # selezioni l'agente corrente
        if action == 0:  # DOC walk
            self.walk(self.learners[agent], agent)
        elif action == 1:  # DOC lay_pheromone
            self.lay_pheromone(self.learners[agent]['pos'], self.lay_amount)
        elif action == 2:  # DOC follow_pheromone
            max_pheromone, max_coords = self._find_max_pheromone(self.learners[agent]['pos'])
            if max_pheromone >= self.sniff_threshold:
                self.follow_pheromone(max_coords, self.learners[agent], agent)
            else:
                self.walk(self.learners[agent], agent)

        self.agent_selection = self._agent_selector.next()  # seleziono l'agente successivo

    # non learners act
    def moving(self):
        # DOC action: 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
        self._evaporate()
        self._diffuse()

        for turtle in self.turtles:
            pos = self.turtles[turtle]['pos']
            t = self.turtles[turtle]
            max_pheromone, max_coords = self._find_max_pheromone(pos)

            if max_pheromone >= self.sniff_threshold:
                self.follow_pheromone(max_coords, t, turtle)
            else:
                self.walk(t, turtle)

            self.lay_pheromone(pos, self.lay_amount)

    # not using .change_all method form BooleanSpace
    def last(self, agent):
        self.agent = agent
        self.obs_dict[self.agent][0] = self._compute_cluster(self.agent) >= self.cluster_threshold
        self.obs_dict[self.agent][1] = self._check_chemical(self.agent)
        cur_reward = self.rewardfunc(self.agent)
        return self.obs_dict[self.agent], cur_reward, False, {}

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
        n_size = len(self.diffuse_patches[list(self.patches.keys())[0]])  # same for every patch
        for patch in self.patches:
            p = self.patches[patch]['chemical']
            ratio = p / n_size
            if p > 0:
                #n_size = len(self.diffuse_patches[patch])
                for n in self.diffuse_patches[patch]:
                    self.patches[n]['chemical'] += ratio
                self.patches[patch]['chemical'] = ratio

    def _evaporate(self):
        """
        :return:
        """
        for patch in self.patches:
            if self.patches[patch]['chemical'] > 0:
                self.patches[patch]['chemical'] *= self.evaporation

    def walk(self, turtle, _id):
        """
        Action 0: move in random direction (8 sorrounding cells
        :param _id: the id of the turtle to move
        :param turtle: the turtle to move
        :return: None (pos is updated after movement as side-effect)
        """
        choice = [self.patch_size, -self.patch_size, 0]
        x, y = turtle['pos']
        self.patches[turtle['pos']]['turtles'].remove(_id)
        x2, y2 = x + np.random.choice(choice), y + np.random.choice(choice)
        x2, y2 = self._wrap(x2, y2)
        turtle['pos'] = (x2, y2)
        self.patches[turtle['pos']]['turtles'].append(_id)

    def follow_pheromone(self, ph_coords, turtle, _id):
        """
        Action 2: move turtle towards greatest pheromone found
        :param _id: the id of the turtle to move
        :param ph_coords: the position where max pheromone has been sensed
        :param turtle: the turtle looking for pheromone
        :return: None (pos is updated after movement as side-effect)
        """
        x, y = turtle['pos']
        self.patches[turtle['pos']]['turtles'].remove(_id)
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
        x, y = self._wrap(x, y)
        turtle['pos'] = (x, y)
        self.patches[turtle['pos']]['turtles'].append(_id)

    def _find_max_pheromone(self, pos):
        """
        Find where the maximum pheromone level is within square 'area' centred in 'pos'
        :param pos: the x,y position of the turtle looking for pheromone
        :return: the maximum pheromone level found and its x,y position
        """
        max_ph = -1
        max_pos = pos
        for p in self.smell_patches[pos]:
            chem = self.patches[p]['chemical']
            if chem > max_ph:
                max_ph = chem
                max_pos = p

        return max_ph, max_pos

    def _compute_cluster(self, agent):
        """
        Checks whether the agent is within a cluster, given 'cluster_radius' and 'cluster_threshold'
        :return: a boolean
        """
        self.agent = agent
        cluster = 1
        for p in self.cluster_patches[self.learners[self.agent]['pos']]:
            cluster += len(self.patches[p]['turtles'])

        return cluster

    def _check_chemical(self, agent):
        """
        Checks whether there is pheromone on the patch where the learner turtle is
        :return: a boolean
        """
        self.agent = agent
        return self.patches[self.learners[self.agent]['pos']][
                   'chemical'] > self.sniff_threshold  # QUESTION should we use self.sniff_threshold here?

    def rewardfunc(self, agent):
        """
        :return: the reward
        """
        self.agent = agent
        cluster = self._compute_cluster(self.agent)
        if cluster >= self.cluster_threshold:
            self.cluster_ticks[self.agent] += 1
            self.cur_reward = 100
        else:
            self.cur_reward = -5
        self.rewards[self.agent].append(self.cur_reward)
        return self.cur_reward

    def reset(self):
        # empty stuff
        self.rewards = {i: [] for i in range(self.learner_population)}
        self.cluster_ticks = {i: 0 for i in range(self.learner_population)}
        self.obs_dict = {a: self.observation_spaces.change_all(False) for a in self.agents}
        # re-position learner turtle
        for l in self.learners:
            self.patches[self.learners[l]['pos']]['turtles'].remove(l)
            self.learners[l]['pos'] = self.coords[np.random.randint(len(self.coords))]
            self.patches[self.learners[l]['pos']]['turtles'].append(l)  # DOC id of learner turtle
        # re-position NON learner turtles
        for t in self.turtles:
            self.patches[self.turtles[t]['pos']]['turtles'].remove(t)
            self.turtles[t]['pos'] = self.coords[np.random.randint(len(self.coords))]
            self.patches[self.turtles[t]['pos']]['turtles'].append(t)
        # patches-own [chemical] - amount of pheromone in the patch
        for p in self.patches:
            self.patches[p]['chemical'] = 0.0

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()

    def render(self, **kwargs):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # chiusura finestra -> termina il programma
                pygame.quit()

        if self.first_gui:
            self.first_gui = False
            pygame.init()
            pygame.display.set_caption("SLIME")

        self.screen.fill(BLACK)
        # disegno le patches
        for p in self.patches:
            chem = round(self.patches[p]['chemical']) * self.shade_strength
            pygame.draw.rect(self.screen, (0, chem if chem <= 255 else 255, 0),
                             pygame.Rect(p[0] - self.offset, p[1] - self.offset, self.patch_size, self.patch_size))
            if self.show_chem_text:
                if self.patches[p]['chemical'] > self.sniff_threshold:
                    text = self.chemical_font.render(str(round(self.patches[p]['chemical'], 1)), True, GREEN)
                    self.screen.blit(text, text.get_rect(center=p))

        # Disegno le turtle learner!
        for l in self.learners.values():
            pygame.draw.circle(self.screen, RED, (l['pos'][0], l['pos'][1]),
                            self.turtle_size // 2)
        # disegno le altre turtles
        for turtle in self.turtles.values():
            pygame.draw.circle(self.screen, BLUE, (turtle['pos'][0], turtle['pos'][1]), self.turtle_size // 2)

        for p in self.patches:
            if len(self.patches[p]['turtles']) > 1:
                text = self.cluster_font.render(str(len(self.patches[p]['turtles'])), True,
                                                RED if -1 in self.patches[p]['turtles'] else WHITE)
                self.screen.blit(text, text.get_rect(center=p))

        self.clock.tick(self.fps)
        pygame.display.flip()

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()


#   MAIN
PARAMS_FILE = "SlimeEnvV2-params.json"
EPISODES = 5
LOG_EVERY = 10

with open(PARAMS_FILE) as f:
    params = json.load(f)
env = Slime(render_mode="human", **params)




# Q-Learning
alpha = 0.1
gamma = 0.6
epsilon = 0.9
decay = 0.95  # di quanto diminuisce epsilon ogni episode

# creo la Q_table
d_qtable = {i: np.zeros([4, 3]) for i in range(params['learner_population'])}


# TRAINING
print("Start training...")
for ep in range(1, 9):
    env.reset()
    print("Epsilon: ", epsilon)
    print(f"-------------------------------------------\nEPISODE: {ep}\n-------------------------------------------")
    for tick in range(400):
        env.moving()
        for agent in env.agent_iter(max_iter=params['learner_population']):
            state, reward, done, info = env.last(agent)  # get observation (state) for current agent

            # converto la mio obs in un numero visto che non posso accedere alla Q Table con una coppia di booleani
            if sum(state) == 0:  # [False, False]
                state = sum(state)  # 0
            elif sum(state) == 2:  # [True, True]
                state = 3
            elif int(state[0]) == 1 and int(state[1]) == 0:  # [True, False] ==> si trova in un cluster ma non su una patch con feromone --> difficile succeda
                state = 1
            else:
                state = 2  # [False, True]

            if random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, 3)  # Explore action space
                # action = env.action_space(agent).sample()                            .action_space not implemented yet
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
    for i in d_qtable:
        print(i, d_qtable[i])
print("Training finished!\n")





"""Evaluate agent's performance after Q-learning"""
for ep in range(1, EPISODES + 1):
    env.reset()
    print(f"-------------------------------------------\nEPISODE: {ep}\n-------------------------------------------")
    for tick in range(params['episode_ticks']):
        env.moving()
        for agent in env.agent_iter(max_iter=params['learner_population']):
            state, rew, done, info = env.last(agent)
            # converto la mio obs in un numero visto che non posso accedere alla Q Table con una coppia di booleani
            if sum(state) == 0:
                state = sum(state)
            elif sum(state) == 2:
                state = 3
            elif int(state[0]) == 1 and int(state[1]) == 0:
                state = 1
            else:
                state = 2

            action = np.argmax(d_qtable[agent][state])
            env.step(action)
        env.render()
env.close()