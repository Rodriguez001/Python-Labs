import math
import random
import sys
import time
from random import randint
import matplotlib.pyplot as plt
import random as rd
import numpy as np
from operator import mul

########## les parametres d'initialisation ###############
#dim = 3   la dimension de la particule
spacewidth = 1  # la dimension du plan de jeu
Vmax = 1

#########################################################
class Particule:

    def __init__(self, dim, p_min, p_max):
        self.vitesse = [0.0 for i in range(dim)]
        self.position = [round(np.random.uniform(p_min, p_max), 3) for i in range(dim)]
        self.pbest = self.position

    def getVitesse(self):
        return self.vitesse

    def getPosition(self):
        return self.position

    def getPbest(self):
        return self.pbest

    def setVitesse(self, acceleration):
        self.vitesse = acceleration

    def setPosition(self, pos):
        self.position = pos

    def setPbest(self, best):
        self.pbest = best



if __name__ == '__main__':
    particule = Particule()


