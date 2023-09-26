# This is a sample Python script.
import math
import sys
import time
from random import random, randint
from Recuit_simule.RS_class import TSP_class


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hello, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# fonction de parsing du fichier des données
def Lire_fichier(filename):
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


def carre(x):
    return x * x


def mat_distances(villes):
    distances = []
    for x in range(len(villes)):
        nvline = []
        for y in range(len(villes)):
            nvline.append(math.sqrt(carre(villes[x][0] - villes[y][0]) + carre(villes[x][1] - villes[y][1])))
            # print('les index de x et y : ', x,y)
        distances.append(nvline)
    return distances


def trajets_aleatoires(villes):
    trajets = []
    k = 1
    while k <= (1000 * len(villes)):
        # trajet = [randint(1, len(villes)) for i in range(len(villes))]
        trajet = []
        i = 1
        while i <= len(villes):
            a = randint(1, len(villes))
            if a not in trajet:
                trajet.append(a)
                i = i + 1
        if trajet not in trajets:
            trajets.append(trajet)
            k = k + 1
    return trajets


def calcul_dist_total(mat_dist, trajet):
    total = mat_dist[trajet[len(trajet) - 1] - 1][trajet[0] - 1]
    if len(trajet) > 1:
        i = 1
        while i < len(trajet):
            total = total + mat_dist[trajet[i - 1] - 1][trajet[i] - 1]
            # print('les index ici : ', trajet[i - 1], trajet[i])
            i = i + 1
    return total


def recherche_naive(mat_dist, liste_trajets):
    if len(liste_trajets) > 0:
        best = liste_trajets[0]
        d = calcul_dist_total(mat_dist, best)
        for trajet in liste_trajets:
            if calcul_dist_total(mat_dist, trajet) < d:
                d = calcul_dist_total(mat_dist, trajet)
                best = trajet
    return best


def closest(ville, mat_dist, visit, liste_villes):
    if len(liste_villes) > 1:
        voisinage = mat_dist[ville]
        min = sys.float_info.max
        indice = 0
        for i in range(len(liste_villes)):
            if (i + 1) not in visit:
                if min > mat_dist[ville][i] != 0:
                    min = mat_dist[ville][i]
                    indice = i
        #(closest, indice) = min((d, indice) for indice, d in enumerate(voisinage))
    return [min, indice]


def hill_climbing(mat_dist, liste_villes):
    if len(liste_villes) > 0:
        results = []
        best = []
        visit = []
        trajet = []
        index_ville = randint(1, len(liste_villes)) - 1
        src = index_ville
        trajet.append(index_ville + 1)
        cpt = 1
        sum_distance = 0
        while cpt <= len(liste_villes):
            if (index_ville + 1) not in visit:
                t = closest(index_ville, mat_dist, visit, liste_villes)
                sum_distance += t[0]
                trajet.append(t[1]+1)
                visit.append(index_ville+1)
                index_ville = t[1]
                cpt += 1
        sum_distance += mat_dist[index_ville][src]
        d = sum_distance
        for rep in range(1000):
            visit = []
            trajet = []
            index_ville = randint(1, len(liste_villes)) - 1
            src = index_ville
            trajet.append(index_ville + 1)
            cpt = 1
            sum_distance = 0
            while cpt < len(liste_villes):
                if (index_ville + 1) not in visit:
                    t = closest(index_ville, mat_dist, visit, liste_villes)
                    sum_distance += t[0]
                    trajet.append(t[1]+1)
                    visit.append(index_ville+1)
                    index_ville = t[1]
                    cpt += 1
            sum_distance += mat_dist[index_ville][src]
            if d > sum_distance:
                d = sum_distance
                best = trajet
    return [trajet, d]

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    rs = TSP_class('RS/22.tsp')
    # mesvilles = Lire_fichier('RS/16.tsp')
    print('mes villes et leurs coordonnées', rs.villes)
    print('la matrice des distance :', rs.matrice)
    # mestrajets = trajets_aleatoires(mesvilles)
    # print('Mes differents trajets:', mestrajets)
    print('le nombre de trajets generés :', len(rs.trajetsAlea))
    tp1 = time.time()
    best_path = rs.recherche_naive()
    tp2 = time.time()
    distance = rs.cout(best_path)
    print('le meilleur trajet est : ', best_path, ' et la distance est : ', distance)
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
