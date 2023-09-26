from random import randint

import fs as fs

from Recuit_simule.RS_class import TSP_class
import time
import numpy as np
import matplotlib.pyplot as plt

########## les parametres d'initialisation ##############
dataSource = 'RS/16.tsp'  # sources de données
T = 1000  # la temperature initiale
Nb_evaluations = 200
#########################################################


if __name__ == '__main__':
    rs = TSP_class(dataSource)
    # rs.plotTowns()
    # mesvilles = Lire_fichier('RS/16.tsp')
    print('mes villes et leurs coordonnées', rs.villes)
    # print('la matrice des distance :', rs.affiche_matrice())
    # mestrajets = trajets_aleatoires(mesvilles)
    # print('Mes differents trajets:', mestrajets)
    print('le nombre de trajets generés :', len(rs.trajetsAlea))
    tp1 = time.time()
    best_path = rs.recherche_naive()
    tp2 = time.time()

    print('le meilleur trajet est pour la recherche naive : ', best_path[0], ' et le cout : ', best_path[1])
    print('le temps mis t1 = ', tp2 - tp1)
    tp1 = time.time()
    result = rs.hill_climbing()
    tp2 = time.time()
    print('le meilleur chemnin pour la recherche maline : ', result[0], ' et le cout : ', result[1])
    print('temps d\'execution t2 = ', tp2 - tp1)
    if result[0] in rs.trajetsAlea:
        print('oui il est bien dans l\' des trajets')
    else:
        print('Non il n\' y est pas du tout')

    f = rs.cout
    X1 = rs.candidat
    stats1 = []
    stats2 = []
    Valeurs_ob_RS = []
    Valeurs_ob_LAHC = []
    for i in range(Nb_evaluations):
        # T = 1000
        #tp1 = time.time()
        result = rs.recuit_simule(f, X1, T)
        #tp2 = time.time()
        #print('le meilleur chemnin pour le recuit simulé : ', result[0], ' et le cout : ', result[1])
        #print('temps d\'execution t3 =  ', tp2 - tp1)

        # T = 1000
        #X2 = rs.candidat
        #tp1 = time.time()
        results = rs.LAHC_TSP(f, X1)
        #tp2 = time.time()
        #print('le meilleur chemnin pour le LAHC : ', results[0], ' et le cout : ', results[1])
        #print('temps d\'execution t4 =  ', tp2 - tp1)
        Valeurs_ob_RS.append(result[1])
        Valeurs_ob_LAHC.append(results[1])
    stats1.append(np.min(Valeurs_ob_RS))
    stats2.append(np.min(Valeurs_ob_LAHC))
    stats1.append(np.max(Valeurs_ob_RS))
    stats2.append(np.max(Valeurs_ob_LAHC))
    stats1.append(np.average(Valeurs_ob_RS))
    stats2.append(np.average(Valeurs_ob_LAHC))
    stats1.append(np.median(Valeurs_ob_RS))
    stats2.append(np.median(Valeurs_ob_LAHC))
    stats1.append(np.std(Valeurs_ob_RS))
    stats2.append(np.std(Valeurs_ob_LAHC))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 6), sharey=True)
    axes[0].boxplot(Valeurs_ob_RS, showmeans=True, meanline=True)
    axes[0].set_title("Recuit simulé", fontsize=12)

    #fig1.show()
    axes[1].boxplot(Valeurs_ob_LAHC, showmeans=True, meanline=True)
    axes[1].set_title("LAHC", fontsize=12)
    fig.suptitle("Comparaison des 2 methodes sur TSP")
    fig.subplots_adjust(hspace=0.4)
    plt.show()
    print("                  statistiques sur les methodes pour ", Nb_evaluations," évaluations : ")
    print(" Criteres    |             RS                 |                  LAHC        |")
    print(" Min(optinum)|       ",stats1[0],"        |       ",stats2[0],"          |")
    print(" Max         |       ", stats1[1], "      |       ", stats2[1], "        |")
    print(" Moyenne     |       ", stats1[2], "      |       ", stats2[2], "        |")
    print(" Mediane     |       ", stats1[3], "      |       ", stats2[3], "        |")
    print(" Ecart-type  |       ", stats1[4], "     |       ", stats2[4], "        |")
