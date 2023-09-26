import random
import time
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
from numpy import product, cos, sqrt, sin

from Particule import Particule

########## les parametres d'initialisation ###############
dimension = 2  # la dimension du problème
N = 200  # le nombre de particules
Nbiter_max = 1000  # le nombre d'iterations
nbiter_cvg = round(Nbiter_max/3)  # le nombre total d'iteration sans convergence
critere_fitness = 1e-3  # critere de fitness
position_min = -1000  # position minimale de l'espace de recherche
position_max = 1000  # position maximale de l'espace de recherche
cmax = 0.7  # coefficient de contrition
tau = 1.74  # coefficient d'inertie
v_max = 1  # Vitesse maximale
v_min = -1  # Vitesse minimale
spacewidth = 1  # l'espace de recherche
fichier_gif = './pso_schwefel.gif'  # le fichier gif de visualisation


#########################################################
class PSO:

    def __init__(self):
        # le nombre de particules
        self.nbparticules = N
        # les positions des particules
        self.particules_obj = self.initialisation(dimension, position_min, position_max)
        self.particules = [p.getPosition() for p in self.particules_obj]
        # les meilleures positions de chaque particule
        self.pbest_position = self.particules
        # le vecteur de fitness ou distance de toutes les positions des particules
        self.pbest_fit = [self.F(p) for p in self.particules]
        # la meilleure position globale
        self.gbest_position = self.pbest_position[np.argmin(self.pbest_fit)]
        # les vitesses des particules commençant à 0.0
        self.vitesses = [p.getVitesse() for p in self.particules_obj]

    def initialisation(self, dim, pos_min, pos_max):
        # print( ' la dim = ', dim)
        # p = Particule(dim)
        # print('une particule (pos, vitesse, best):', p.getPosition(), p.getVitesse(), p.getPbest())
        return [Particule(dim, pos_min, pos_max) for i in range(self.nbparticules)]

    def critereDarret(self, cpt1, cpt2):
        return (cpt2 <= nbiter_cvg) \
            and (cpt1 <= Nbiter_max) \
            and (np.average(self.pbest_fit) > critere_fitness)

    def F(self, p):
        return 418.9829 * len(p) - sum([x * sin(sqrt(np.abs(x))) for x in p])
        # reduce(lambda x, a: a + x ** 2, p)

    def bornage(self, particule, speed, pos_min, pos_max, speed_min, speed_max):
        pos = particule[:]
        vit = speed[:]

        for i in range(len(particule)):
            if pos[i] < pos_min:
                pos[i] = pos_min
                vit[i] *= np.random.uniform(-1, 0)
            if pos[i] > pos_max:
                pos[i] = pos_max
                vit[i] *= np.random.uniform(-1, 0)
            if vit[i] < speed_min:
                vit[i] = speed_min
            if vit[i] > speed_max:
                vit[i] = speed_max
        return pos, vit

    def OEP(self):
        f = self.F
        compteur = 0
        cmp_maj = 0
        # Plotting prepartion
        fig1, ax1 = plt.subplots()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        x = np.linspace(position_min, position_max, 80)
        y = np.linspace(position_min, position_max, 80)
        X, Y = np.meshgrid(x, y)
        Z = self.F([X, Y])
        ax.plot_wireframe(X, Y, Z, color='r', linewidth=0.2)

        # Animation image placeholder
        images = []
        globaux = []
        bool = False
        while self.critereDarret(compteur, cmp_maj):
            # self.show_particules(self.particules)
            globaux.append(f(self.gbest_position))
            particles = self.particules
            # Add plot for each generation (within the generation for-loop)
            image = ax.scatter3D([
                particles[n][0] for n in range(self.nbparticules)],
                [particles[n][1] for n in range(self.nbparticules)],
                [self.F(particles[n]) for n in range(self.nbparticules)], c='b')
            images.append([image])

            compteur += 1
            # mise a jour de sa meilleur position
            # mise a jour de la meilleure position du voisinage

            for i in range(self.nbparticules):
                result = self.majlocal(i, 3, self.pbest_fit)
                vbest = result[1]
                # print('particule : ', i, ' ', self.particules[i])
                if f(vbest) < f(self.pbest_position[i]):
                    self.pbest_position[i] = vbest[:]
                    self.pbest_fit[i] = f(vbest)
                    bool = True
                if f(self.pbest_position[i]) < f(self.gbest_position):
                    self.gbest_position = (self.pbest_position[i])[:]
                    self.pbest_fit[i] = f(self.pbest_position[i])
                    # print('gbest :', self.gbest_position)
                    bool = True
            # Deplacement des particules
            if bool:
                cmp_maj += 1
            # mise a jour des vitesses et des positions
            for i in range(self.nbparticules):
                result = self.majlocal(i, 3, self.pbest_fit)
                vbest = result[1]
                self.vitesses[i] = self.speedup(self.particules[i],
                                                self.vitesses[i],
                                                self.pbest_position[i],
                                                self.gbest_position,
                                                vbest)
                self.particules[i] = self.update_position(self.particules[i], self.vitesses[i])
                # on applique les mesures de contraintes sur les bornes
                res = self.bornage(self.particules[i], self.vitesses[i], position_min, position_max, v_min, v_max)
                self.particules[i] = res[0]
                self.vitesses[i] = res[1]
        # Generate the animation image and save
        animated_image = animation.ArtistAnimation(fig, images)
        animated_image.save(fichier_gif, writer='pillow')
        x1 = np.arange(0, compteur, 1)
        line, = ax1.plot(x1, globaux)
        ax1.set_xlabel('itérations')
        ax1.set_ylabel('optimum global')
        plt.show()
        # liste[i] = self.bornage(liste[i])
        return [self.gbest_position, compteur, min(self.pbest_fit)]

    def majlocal(self, i, nbv, liste_best_obj):
        n = round((nbv - 1) / 2)
        r = (nbv - 1) - n
        index = []
        longueur = len(liste_best_obj)
        for j in sorted(range(n), reverse=True):
            if i - j < 0:
                index.append(i - j + longueur)
            else:
                index.append(i - j)
        index.append(i)
        for j in range(r):
            if i + j > longueur - 1:
                index.append(i + j - longueur)
            else:
                index.append(i + j)
        best = np.max(liste_best_obj)
        index_retour = 0
        for a in range(nbv):
            if liste_best_obj[index[a]] < best:
                best = liste_best_obj[index[a]]
                index_retour = index[a]
        return [best, self.particules[index_retour]]

    def speedup(self, particle, velocity, pbest, gbest, vbest, w_min=0.5, max=1.0, c=0.1):
        # nouvelle vitesse
        num_particule = len(particle)
        new_speed = [0.0 for i in range(num_particule)]
        # on genere aleatoirement les composantes
        r1 = random.uniform(0, max)
        r2 = random.uniform(0, max)
        r3 = random.uniform(0, max)
        w = random.uniform(w_min, max)
        c1 = c
        c2 = c
        c3 = c
        # Calcul de la nouvelle vitesse
        for i in range(num_particule):
            new_speed[i] = w * velocity[i] + c1 * r1 * (pbest[i] - particle[i]) + c2 * r2 * (gbest[i] - particle[i]) \
                           + c3 * r3 * (vbest[i] - particle[i])
        return new_speed

    def update_position(self, particule, speed):
        return [p + s for p, s in zip(particule, speed)]

    def show_particules(self, particles):
        # Plotting prepartion
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        x = np.linspace(position_min, position_max, 80)
        y = np.linspace(position_min, position_max, 80)
        X, Y = np.meshgrid(x, y)
        Z = self.F([X, Y])
        ax.plot_wireframe(X, Y, Z, color='r', linewidth=0.3)

        # Animation image placeholder
        images = []

        # Add plot for each generation (within the generation for-loop)
        image = ax.scatter3D([
            particles[n][0] for n in range(self.nbparticules)],
            [particles[n][1] for n in range(self.nbparticules)],
            [self.F(particles[n]) for n in range(self.nbparticules)], c='b')
        images.append([image])

        # Generate the animation image and save
        animated_image = animation.ArtistAnimation(fig, images)
        animated_image.save(fichier_gif, writer='pillow')
        # plt.show()
        # animated_image.resume()


if __name__ == '__main__':
    pso = PSO()
    print('le nombre de particules : ', pso.nbparticules)
    print('liste des particules : ', pso.particules)

    tp1 = time.time()
    resultat = pso.OEP()
    tp2 = time.time()
    print('temps d\'execution : ', tp2 - tp1)
    print('la meilleure position globale :', resultat[0])
    print('la meilleure distance ou fitness: ', resultat[2])
    print('Moyenne de differentiel des distances des particules: ', np.average(pso.pbest_fit))
    print('Le nombre d\'iterations pour la convergence : ', resultat[1])

#    pso.OEP()
