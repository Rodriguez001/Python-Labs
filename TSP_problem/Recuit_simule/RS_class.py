import math
import sys
import time
from random import random, randint
import matplotlib.pyplot as plt
import random as rd

import numpy as np

########## les parametres d'initialisation ###############
alpha = 0.9  # la facteur de refroidissement
paliers = 20  # iterations d'équilibrethermodynamique
T_mini = 1E-2  # la temperature minimale

Nb_iterations = 10000  # le nombre max d'itérations de la primitive
N_limit = 1000  # le nombre max sans amélioration l'optimum
Taille_buffer = 100  # Taille de la mémoire ou du buffer
Nb_sacs_aleatoires = 1000  # le nombre d'essais générés aleatoirement
#########################################################
class TSP_class:

    def __init__(self, data):
        self.villes = self.Lire_fichier(data)
        self.trajetsAlea = self.trajets_aleatoires()
        self.matrice = self.mat_distances()
        self.candidat = self.firstCandidate()

    def Lire_fichier(self, filename):
        fichier = open(filename, 'r')
        lignes = fichier.readlines()
        villes = []
        i = 0
        for ligne in lignes:
            tmp = ligne.split()
            # print('ligne :', i, tmp[1], tmp[2])
            villes.append([float(tmp[1]), float(tmp[2])])
            i = i + 1
            # lignes = fichier.readline()
        return villes

    def carre(self, x):
        return x * x

    def mat_distances(self):
        distances = []
        for x in range(len(self.villes)):
            nvline = []
            for y in range(len(self.villes)):
                nvline.append(np.sqrt(self.carre(self.villes[x][0] - self.villes[y][0]) + self.carre(
                    self.villes[x][1] - self.villes[y][1])))
                # print('les index de x et y : ', x,y)
            distances.append(nvline)
        return distances

    def affiche_matrice(self):
        for i in range(len(self.villes)):
            print(self.matrice[i])

    def plotTowns(self):
        x = []
        y = []
        for i in range(len(self.villes)):
            x.append(self.villes[i][0])
        for i in range(len(self.villes)):
            y.append(self.villes[i][1])
        print('les abcisses : ', x)
        print('les ordonnées : ', y)
        # plt.plot(x+[x[0]], y+[y[0]], "o-")
        plt.plot(x, y, "o")
        plt.show()

    def firstCandidate(self):
        trajet = []
        i = 1
        while i <= len(self.villes):
            a = randint(1, len(self.villes))
            if a not in trajet:
                trajet.append(a)
                i = i + 1
        return trajet

    def trajets_aleatoires(self):
        trajets = []
        k = 1
        while k <= (1000 * len(self.villes)):
            # trajet = [randint(1, len(villes)) for i in range(len(villes))]
            trajet = []
            i = 1
            while i <= len(self.villes):
                a = randint(1, len(self.villes))
                if a not in trajet:
                    trajet.append(a)
                    i = i + 1
            if trajet not in trajets:
                trajets.append(trajet)
                k = k + 1
        return trajets

    def cout(self, trajet):
        total = self.matrice[trajet[len(trajet) - 1] - 1][trajet[0] - 1]
        if len(trajet) > 1:
            i = 1
            while i < len(trajet):
                total = total + self.matrice[trajet[i - 1] - 1][trajet[i] - 1]
                # print('les index ici : ', trajet[i - 1], trajet[i])
                i = i + 1
        return total

    def recherche_naive(self):
        if len(self.trajetsAlea) > 0:
            best = self.trajetsAlea[0]
            d = self.cout(best)
            for trajet in self.trajetsAlea:
                if self.cout(trajet) < d:
                    d = self.cout(trajet)
                    best = trajet
        dist = self.cout(best)
        return [best, dist]

    def closest(self, ville, visit):
        if len(self.villes) > 1:
            voisinage = self.matrice[ville]
            min = sys.float_info.max
            indice = 0
            for i in range(len(self.villes)):
                if (i + 1) not in visit:
                    if min > self.matrice[ville][i] != 0:
                        min = self.matrice[ville][i]
                        indice = i
            # (closest, indice) = min((d, indice) for indice, d in enumerate(voisinage))
        return [min, indice]

    def hill_climbing(self):
        if len(self.villes) > 0:
            results = []
            best = []
            visit = []
            trajet = []
            # index_ville = randint(1, len(self.villes)) - 1
            index_ville = 0
            src = index_ville
            trajet.append(index_ville + 1)
            cpt = 1
            sum_distance = 0
            while cpt <= len(self.villes):
                if (index_ville + 1) not in visit:
                    t = self.closest(index_ville, visit)
                    sum_distance += t[0]
                    trajet.append(t[1] + 1)
                    visit.append(index_ville + 1)
                    index_ville = t[1]
                    cpt += 1
            sum_distance += self.matrice[index_ville][src]
            d = sum_distance
            for rep in range(1000):
                visit = []
                trajet = []
                index_ville = randint(1, len(self.villes)) - 1
                src = index_ville
                trajet.append(index_ville + 1)
                cpt = 1
                sum_distance = 0
                while cpt < len(self.villes):
                    if (index_ville + 1) not in visit:
                        t = self.closest(index_ville, visit)
                        sum_distance += t[0]
                        trajet.append(t[1] + 1)
                        visit.append(index_ville + 1)
                        index_ville = t[1]
                        cpt += 1
                sum_distance += self.matrice[index_ville][src]
                if d > sum_distance:
                    d = sum_distance
                    best = trajet
        return [trajet, d]

    def permutation(self, X, i, j):
        if len(X) != 0:
            tmp = X[i]
            X[i] = X[j]
            X[j] = tmp
        return X

    def perturbation(self, X):
        X_voisin = X[:]
        i = randint(0, len(X) - 1)
        j = randint(0, len(X) - 1)
        while i == j:
            i = randint(0, len(X) - 1)
            j = randint(0, len(X) - 1)
        min_p = np.min([i, j])
        max_p = np.max([i, j])
        # X_voisin = self.permutation(X_voisin, i, j)
        X_voisin[min_p:max_p] = X_voisin[min_p:max_p].copy()[::-1]
        return X_voisin

    def critereMetropolis(self, delta, Temperature):
        if delta <= 0:
            return True
        else:
            return rd.uniform(0, 1) < np.exp(-delta / Temperature)

    def critereConvergence(self, compteur, a, b):
        if compteur > 1000000 and (a >= 0 or b >= 0):
            print('Stop pour cause de non convergence')
            return True
        else:
            return False

    def equilibreThermodynamique(self):
        return True

    def refroidissement(self, Temperature):
        "refroidissement par la méthode géométrique"
        # tau = 1E3
        # alpha = np.exp(-1*t/tau)
        # alpha = 0.99
        return alpha * Temperature

    def recuit_simule(self, f, X, Temperature):
        # N = 30
        X_min = X[:]
        f_min = f(X)
        T_min = T_mini
        f_X = f(X)
        T = Temperature
        k = 0
        # X_vois = self.perturbation(X)
        # delta_f = f(X_vois) - f_X
        # cpt = 0
        while T > T_min:  # and not(self.critereConvergence(cpt, delta_f, f(X_vois)-f_min)):
            # while not(self.equilibreThermodynamique()):
            # print('temperature : ', T)
            for i in range(paliers):
                X_vois = self.perturbation(X)
                delta_f = f(X_vois) - f_X
                if self.critereMetropolis(delta_f, T):
                    X = X_vois[:]
                    f_X = f(X_vois)
                    if delta_f < 0 and f(X_vois) < f_min:
                        f_min = f(X_vois)
                        X_min = X_vois[:]
                        #print(" Amélioration, ici compteur : ", k)
                        cpt = 0
                # cpt += 1
                # print('nouveau min : ', X_min, ' et son cout est : ', f_min, ' - ', f(X_min))
            k += 1
            # print('nouveau min : ', X_min, ' et son cout est :  ', f_min, ' le compteur : ', k)
            T = self.refroidissement(T)
        #print('le nombre d\'iterations = ', k)
        return [X_min, f_min]

    def LAHC_TSP(self, f, X):
        f_min = f(X)
        X_min = X[:]
        buffer = [f_min for i in range(Taille_buffer)]
        compteur = 0
        compteur2 = 0
        while compteur <= Nb_iterations:
            X_vois = self.perturbation(X)
            f_mem = buffer[compteur % Taille_buffer]
            if f_mem >= f(X_vois):
                X = X_vois[:]
            buffer[compteur % Taille_buffer] = f(X)
            if f_min > f(X_vois):
                X_min = X_vois[:]
                f_min = f(X_vois)
                compteur2 = compteur
            compteur += 1
        #print('le nombre d\'iterations = ', compteur2)
        return [X_min, f_min]



if __name__ == '__main__':
    rs = TSP_class('../RS/22.tsp')
    # mesvilles = Lire_fichier('RS/16.tsp')
    print('mes villes et leurs coordonnées', rs.villes)
    print('la matrice des distance :', rs.matrice)
    # mestrajets = trajets_aleatoires(mesvilles)
    # print('Mes differents trajets:', mestrajets)
    print('le nombre de trajets generés :', len(rs.trajetsAlea))
    tp1 = time.time()
    best_path = rs.recherche_naive()
    tp2 = time.time()
    distance = rs.cout(best_path[0])
    print('le meilleur trajet est : ', best_path[0], ' et la distance est : ', distance)
    print('le temps mis : ', tp2 - tp1)
    tp1 = time.time()
    result = rs.hill_climbing()
    tp2 = time.time()
    print('le meilleur chemnin : ', result[0], ' pour une distance de : ', result[1])
    print('temps d\'execution : ', tp2 - tp1)
    if result[0] in rs.trajetsAlea:
        print('oui il est bien dans l\' des trajets')
    else:
        print('Non il n\' y est pas du tout')
    rs.plotTowns()
