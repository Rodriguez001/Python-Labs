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
input_data = 'data/ks_19_0'  # Source des données
Nb_iterations = 2000  # le nombre max d'itérations de la primitive
N_limit = 500  # le nombre max sans améliorer l'optimum
Taille_buffer = 20  # Taille de la mémoire ou du buffer
Nb_sacs_aleatoires = 100  # le nombre d'essais générés aleatoirement


#########################################################
class KnapsackClass:

    def __init__(self):
        self.items_number = 0
        self.items = []
        self.poids_max = 0
        self.values = []
        self.weights = []
        self.Lire_fichier(input_data)
        self.sacsAlea = self.Random_Solver()
        self.candidat = self.firstCandidate()
        self.best = []
        self.voisinage = []

    def Lire_fichier(self, filename):
        fichier = open(filename, 'r')
        lignes = fichier.readlines()
        items = []
        i = 0
        for ligne in lignes:
            tmp = ligne.split()
            if i == 0:
                self.items_number = int(tmp[0])
                self.poids_max = int(tmp[1])
            else:
                # print('ligne :', i, tmp[1], tmp[2])
                self.items.append([int(tmp[0]), int(tmp[1])])
                self.values.append(int(tmp[0]))
                self.weights.append(int(tmp[1]))
            i = i + 1
            # lignes = fichier.readline()

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
        return [randint(0, 1) for i in range(self.items_number)]

    def Random_Solver(self):
        sacs = []
        k = 1
        # On verifie si l'on ne genere pas plus que le nombre possible total de cas
        if pow(2, self.items_number) > Nb_sacs_aleatoires * self.items_number:
            n = Nb_sacs_aleatoires * self.items_number
        else:
            n = pow(2, self.items_number)
        while k <= n:
            sac_essai = [randint(0, 1) for i in range(self.items_number)]
            while sac_essai in sacs:
                sac_essai = [randint(0, 1) for i in range(self.items_number)]
            k += 1
            sacs.append(sac_essai)
        return sacs

    def cout(self, sac):
        # return sum([a * b for (a, b) in zip(self.values, sac)])
        f = lambda lst1, lst2: sum(map(mul, lst1, lst2))
        return f(self.values, sac)

    def poids(self, sac):
        # return sum([a * b for (a, b) in zip(self.values, sac)])
        f = lambda lst1, lst2: sum(map(mul, lst1, lst2))
        return f(self.weights, sac)

    def checkConstraint(self, sac):
        return self.poids(sac) <= self.poids_max

    def recherche_naive(self):
        # Meilleur_sec = lambda L: np.max(map(self.cout(L)))
        beau_sac = []
        if len(self.sacsAlea) > 0:
            d = 0
            for sac in self.sacsAlea:
                # print('le sac : ', sac,' valeur : ', self.cout(sac),' poids du sac', self.poids(sac))
                if self.cout(sac) > d and self.checkConstraint(sac):
                    d = self.cout(sac)
                    beau_sac = sac[:]
            optimum = self.cout(beau_sac)
            if self.cout(self.best) < optimum:
                self.best = beau_sac[:]
        return optimum

    def getValues(self, liste):
        return liste[0]

    def getPoids(self, liste):
        return liste[1]

    def densite(self, liste):
        return liste[0] / liste[1]

    def tri_Par_densite(self):
        return sorted(self.items, key=self.densite, reverse=True)

    def tri_Par_Valeur(self):
        return sorted(self.items, key=self.getValues)

    def tri_Par_Poids(self, ordre):
        if ordre == 0:
            return sorted(self.items, key=self.getPoids)
        else:
            return sorted(self.items, key=self.getPoids, reverse=True)

    def getPositions_initiales(self, Liste):
        X = []
        for i in range(self.items_number):
            X.append(self.items.index(Liste[i]))
        return X

    def recherche_tri_valeur(self):
        Tampon = self.tri_Par_Valeur()
        positions = self.getPositions_initiales(Tampon)
        # print('Apres le tri : ', Tampon)
        X = []
        som = 0
        poids = 0
        for i in range(self.items_number):
            som += Tampon[i][0]
            poids += Tampon[i][1]
            if poids > self.poids_max:
                som -= Tampon[i][0]
                poids -= Tampon[i][1]
                X.append(0)
            else:
                X.append(1)
        Y = [0 for i in range(self.items_number)]
        for j in range(self.items_number):
            Y[positions[j]] = X[j]
        optimum = self.cout(Y)
        if self.cout(self.best) < optimum:
            self.best = Y[:]
        return [Y, som, poids]

    def recherche_tri_Poids_asc_desc(self, ordre):
        Tampon = self.tri_Par_Poids(ordre)
        positions = self.getPositions_initiales(Tampon)
        # print('Apres le tri : ', Tampon)
        X = []
        som = 0
        poids = 0
        for i in range(self.items_number):
            som += Tampon[i][0]
            poids += Tampon[i][1]
            if poids > self.poids_max:
                som -= Tampon[i][0]
                poids -= Tampon[i][1]
                X.append(0)
            else:
                X.append(1)
        Y = [0 for i in range(self.items_number)]
        for j in range(self.items_number):
            Y[positions[j]] = X[j]
        optimum = self.cout(Y)
        if self.cout(self.best) < optimum:
            self.best = Y[:]
        return [Y, som, poids]

    def recherche_tri_densite(self):
        Tampon = self.tri_Par_densite()
        positions = self.getPositions_initiales(Tampon)
        # print('Apres le tri : ', Tampon)
        X = []
        som = 0
        poids = 0
        for i in range(self.items_number):
            som += Tampon[i][0]
            poids += Tampon[i][1]
            if poids > self.poids_max:
                som -= Tampon[i][0]
                poids -= Tampon[i][1]
                X.append(0)
            else:
                X.append(1)
        Y = [0 for i in range(self.items_number)]
        for j in range(self.items_number):
            Y[positions[j]] = X[j]
        optimum = self.cout(Y)
        if self.cout(self.best) < optimum:
            self.best = Y[:]
        return [Y, som, poids]

    def low_cost(self, X, code):
        ind = 0
        if len(X) != 0:
            if code == 0:
                # f = lambda l1, l2: min(map(mul, l1, l2))
                # val = f(self.values, X)
                # print(" la valeur du petit : ", val)
                min_value = 1E100
                for i in range(self.items_number):
                    if X[i] == 1 and self.values[i] < min_value:
                        min_value = self.values[i]
                        ind = i
                # print(" la valeur du petit : ", min_value)
            else:
                min_value = 1E100
                for i in range(self.items_number):
                    if X[i] == 0 and self.values[i] < min_value:
                        min_value = self.values[i]
                        ind = i
                # print(" la valeur du petit : ", min_value)

        return ind

    def genereVoisin(self, X, cmp):
        X_voisin = X[:]
        # print(" un voisin generé a l'entree : ", X_voisin)
        if cmp == 0 and self.checkConstraint(X_voisin):
            return X_voisin
        if min(X_voisin) == 0:
            i = randint(0, len(X_voisin) - 1)
            while X_voisin[i] == 1:
                i = randint(0, len(X_voisin) - 1)
            X_voisin[i] = 1
        # print(" un voisin generé avant arrangement : ", X_voisin)
        while not self.checkConstraint(X_voisin):
            # print(" position changee : ", self.low_cost(X_voisin))
            X_voisin[self.low_cost(X_voisin, 0)] = 0
        return X_voisin

    def critereConvergence(self, compteur, cpt, cond_maj):
        if compteur > Nb_iterations or (cpt > N_limit and cond_maj < 0):
            print('Stop pour cause de non convergence')
            return True
        else:
            return False

    ## Late acceptance hill climbing ( Maximisation du problème du Sac à dos)
    def LAHC(self, f, X):
        "Late acceptance hill climbing ( Maximisation du problème du Sac à dos)"""
        print('le candidat de depart : ', X)

        compteur = 0
        X_vois = self.genereVoisin(X, compteur)
        # print('le 1er voisin du candidat de depart : ', X_vois)
        X = X_vois[:]
        X_max = X[:]
        f_max = f(X)
        buffer = [f_max for i in range(Taille_buffer)]
        # print('le buffer : ', buffer)

        f_mem = buffer[compteur % Taille_buffer]

        delta_f = f(X_vois) - f_mem
        cpt = 0
        while not (self.critereConvergence(compteur, cpt, f(X_vois) - f_max)):
            # print('je suis dans la boucle')
            X_vois = self.genereVoisin(X, compteur)
            # print(" un voisin generé : ", X_vois)
            f_mem = buffer[compteur % Taille_buffer]
            delta_f = f(X_vois) - f_mem
            if delta_f > 0:
                X = X_vois[:]
                # print(" nouvelle valeur de X : ", X)
            buffer[compteur % Taille_buffer] = f(X)
            if f(X_vois) > f_max:
                f_max = f(X_vois)
                X_max = X_vois[:]
                cpt = 0
                # print(" Amélioration du sac : ", X_max, ' compteur = ', compteur)
            cpt += 1
            # print('nouveau max : ', X_max, ' et son cout est : ', f_max, ' et le compteur = ', compteur)
            compteur += 1
        print('le nombre d\'iterations = ', compteur)
        if self.cout(self.best) < f_max:
            self.best = X_max[:]
        return [X_max, f_max, self.poids(X_max)]


if __name__ == '__main__':
    kp = KnapsackClass()
    # mesvilles = Lire_fichier('RS/16.tsp')
    print('la liste des elements : ', kp.items)
    print('le nombre d\'elements total :', kp.items_number)
    print('le nombre de sacs aléatoirement generés :', len(kp.sacsAlea))
    # print('les sacs aleatoires : ', kp.sacsAlea)
    tp1 = time.time()
    best_value = kp.recherche_naive()
    tp2 = time.time()
    print('le meilleur sac pour la recherche naive : ', kp.best, ' et sa valeur totale est : ', best_value,
          ' son poids total :', kp.poids(kp.best))
    print('le temps mis : ', tp2 - tp1)
    tp1 = time.time()
    best = kp.recherche_tri_valeur()
    tp2 = time.time()
    print('le meilleur sac pour la recherche avec tri de valeurs : ', best[0], ' et sa valeur totale est : ', best[1],
          ' son poids total :', best[2])
    print('le temps mis : ', tp2 - tp1)
    tp1 = time.time()
    best = kp.recherche_tri_Poids_asc_desc(0)
    tp2 = time.time()
    print('le meilleur sac pour la recherche avec tri croissant des poids : ', best[0], ' et sa valeur totale est : ',
          best[1],
          ' son poids total :', best[2])
    print('le temps mis : ', tp2 - tp1)
    tp1 = time.time()
    best = kp.recherche_tri_Poids_asc_desc(1)
    tp2 = time.time()
    print('le meilleur sac pour la recherche avec tri decroissant des poids : ', best[0], ' et sa valeur totale est : ',
          best[1],
          ' son poids total :', best[2])
    print('le temps mis : ', tp2 - tp1)
    tp1 = time.time()
    best = kp.recherche_tri_densite()
    tp2 = time.time()
    print('le meilleur sac pour la recherche avec tri par densité des elements : ', best[0],
          ' et sa valeur totale est : ',
          best[1],
          ' son poids total :', best[2])
    print('le temps mis : ', tp2 - tp1)
    f = kp.cout
    tp1 = time.time()
    best = kp.LAHC(f, kp.candidat)
    tp2 = time.time()
    print('le meilleur sac pour la recherche avec LAHC : ', best[0],
          ' et sa valeur totale est : ',
          best[1],
          ' son poids total :', best[2])
    print('le temps mis : ', tp2 - tp1)

    # rs.plotTowns()
