import gym
import pygame
from gym import spaces
import numpy as np

width = height = 500  # SUPPONGO CHE LA GRIGLIA SIA SEMPRE UN QUADRATO
population = 500


class Slime(gym.Env):

    def __init__(self):

        self.sniff_threshold = 12

        self.step = 5

        # create learner turtle
        self.cord_learner_turtle = []
        self.cord_learner_turtle.append(np.random.randint(10, width - 10))
        self.cord_learner_turtle.append(np.random.randint(10, height - 10))

        # create NON learner turtle
        self.cord_non_learner_turtle = {}
        for p in range(population):
            self.l = []
            self.l.append(np.random.randint(10, width-10))
            self.l.append(np.random.randint(10, height-10))
            self.cord_non_learner_turtle[str(p)] = self.l

        # patches-own [chemical] - amount of pheromone in the patch
        self.chemicals_level = {}
        for x in range(width + 1):
            for y in range(height + 1):
                self.chemicals_level[str(x) + str(y)] = 0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Dict = {}  # da finire

    def moving_turtle(self, action):

        for turtle in self.cord_non_learner_turtle:
            self.max_lv = 0
            self.cord_max_lv = []
            self.bonds = []
            # self.start_x = self.cord_non_learner_turtle[turtle][0] - 3
            # self.start_y = self.cord_non_learner_turtle[turtle][1] - 3
            # self.end_x = self.cord_non_learner_turtle[turtle][0] + 4
            # self.end_y = self.cord_non_learner_turtle[turtle][1] + 4
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
                    pass # allora il punto è dove mi trovo quindi stò fermo (?)
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

        # EVAPORATE CHEMICAL
        for patch in self.chemicals_level:
            if self.chemicals_level[patch] != 0:
                self.chemicals_level[patch] -= 2

    def reset(self):
        # create learner turtle
        self.cord_learner_turtle = []
        self.cord_learner_turtle.append(np.random.randint(10, width - 10))
        self.cord_learner_turtle.append(np.random.randint(10, height - 10))

        # create NON learner turtle
        self.cord_non_learner_turtle = {}
        for p in range(population):
            self.l = []
            self.l.append(np.random.randint(10, width - 10))
            self.l.append(np.random.randint(10, height - 10))
            self.cord_non_learner_turtle[str(p)] = self.l

        # patches-own [chemical] - amount of pheromone in the patch
        self.chemicals_level = {}
        for x in range(width + 1):
            for y in range(height + 1):
                self.chemicals_level[str(x) + str(y)] = 0

    def render(self):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.screen.fill((0, 0, 0))
        self.clock = pygame.time.Clock()
        self.clock.tick(15)  # FPS = 30 lag

        # Disegno LA turtle learner!
        pygame.draw.circle(self.screen, (190, 0, 0), (self.cord_learner_turtle[0], self.cord_learner_turtle[1]), 3)

        for turtle in self.cord_non_learner_turtle.values():
            pygame.draw.circle(self.screen, (0, 190, 0), (turtle[0], turtle[1]), 3)
        pygame.display.flip()





#####################
####### MAIN ########
#####################

env = Slime()
episodes = 5
ticks_per_episode = 700
for ep in range(episodes):
    print(f"EPISODE: {ep}")
    env.reset()
    env.render()
    for tick in range(ticks_per_episode):
        action = np.random.randint(3)
        env.moving_turtle(action)
        env.render()