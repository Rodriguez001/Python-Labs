# les operations de convolutions.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from math import exp


# On crée d'abord d'abord l'image et le motif
def run():
    taille_image = int(input("Donnez la taille de l'image"))
    taille_motif = taille_image
    while (taille_motif >= taille_image):
        taille_motif = int(input("Donnez la taille du motif"))
    taille_conv = taille_image - taille_motif + 1
    max_val = 255
    image = np.randon.randint(max_val+1, size=(taille_image, taille_image))
    motif = np.randon.randint(max_val + 1, size=(taille_image, taille_image))



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    run()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Rodrigue')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
