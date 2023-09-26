import numpy as ql


class Bellman:

    def __init__(self):
        self.Matrix = self.Matrice()

    def Matrice(self):
        P = ql.matrix(ql.zeros([7, 7]))
        return P

    def display_Matrix(self):
        print(self.Matrix)

if __name__ == '__main__':
    B = Bellman()
    B.display_Matrix()
