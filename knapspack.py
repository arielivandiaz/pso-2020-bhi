
from params import params

USE_JAX = 0
if (USE_JAX):
    import jax.numpy as np
    from jax import random
    from jax import grad, jit
else:
    import numpy as np


class Knapspack:
    def __init__(self):
        self.pv = np.zeros(params.items)
        self.rs = np.zeros((params.items, params.rest))
        self.bs = np.zeros(params.rest)
        self.generar_instancia()

    def generar_instancia(self):

        if (USE_JAX):
            key = random.PRNGKey(0)

        alfa = 0.75

        if (USE_JAX):
            self.rs = random.randint(
                key, (params.rest, params.items), 0, 1000)
            q = random.uniform(key, (params.items, 1), None)
        else:
            self.rs = np.random.randint(1000, size=(
                params.rest, params.items))
            q = np.random.random_sample((params.items))

        self.bs = alfa * np.sum(self.rs, axis=1)

        self.pv = (1 / params.rest) * np.sum(self.rs, axis=0)
        self.pv += 500*q

    def mochila(self, X):

        P = len(X)
        N = len(X[0])
        M = params.rest

        Peso = np.sum(self.pv, axis=0)

        PR = np.zeros(P)
        G = np.zeros((P, M))
        VIO = np.zeros((P, M))
        PNLTY = np.zeros(P)
        FI = np.zeros(P)

        for i in range(P):
            PR[i] = 0.0
            for j in range(N):
                PR[i] = self.pv[j] * X[i, j] + PR[i]  # Funcion objetivo

            for k in range(M):
                G[i, k] = 0.0
                for j in range(N):
                    G[i, k] = self.rs[k, j] * X[i, j] + \
                        G[i, k]  # Restricciones
                VIO[i, k] = G[i, k] - self.bs[k]  # Violaciones

            PNLTY[i] = 0.0

            for k in range(M):
                if (VIO[i, k] <= 0.0):
                    VIO[i, k] = 0.0
                PNLTY[i] = VIO[i, k] + PNLTY[i]  # Penalidad

            FI[i] = PR[i] - Peso * PNLTY[i]  # Func.Obj.con penalizacion

        return -FI

    def mochila_res(self, X):

        P = len(X)
        N = len(X[0])
        M = params.rest

        Peso = np.sum(self.pv, axis=0)

        PR = np.zeros(P)
        G = np.zeros((P, M))
        VIO = np.zeros((P, M))
        PNLTY = np.zeros(P)
        FI = np.zeros(P)

        for i in range(P):
            PR[i] = 0.0
            for j in range(N):
                PR[i] = self.pv[j] * X[i, j] + PR[i]  # Funcion objetivo

            for k in range(M):
                G[i, k] = 0.0
                for j in range(N):
                    G[i, k] = self.rs[k, j] * X[i, j] + \
                        G[i, k]  # Restricciones
                VIO[i, k] = G[i, k] - self.bs[k]  # Violaciones

            PNLTY[i] = 0.0

            for k in range(M):
                if (VIO[i, k] <= 0.0):
                    VIO[i, k] = 0.0
                PNLTY[i] = VIO[i, k] + PNLTY[i]  # Penalidad

            FI[i] = PR[i] - Peso * PNLTY[i]  # Func.Obj.con penalizacion

        return FI, VIO

    def check_vio(self, X):

        solPru = np.random.randint(2, size=(1, params.items))
        for k in range(0, params.items):
            solPru[0, k] = X[k]
        [f, vio] = self.mochila_res(solPru)

        return np.count_nonzero(vio)
