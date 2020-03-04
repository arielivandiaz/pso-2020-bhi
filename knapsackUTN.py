import PSOjax
import functions as fxJAX
import functions as fx
import PSOpyswarms as PSOps
from common import *
import time
import numpy as np
import pyswarms as ps
import random
import PSO


def mochila(posibles_soluciones):

    
    particulas = len(posibles_soluciones)
    dimensiones = len(posibles_soluciones[0])

    # Parametros

    reconpensa_por_item = np.array(
        [100.0, 600.0, 1200.0, 2400.0, 500.0, 2000.0])
    restriccion_peso_max = np.array(
        [80.0, 96.0, 20.0, 36.0, 44.0, 48.0, 10.0, 18.0, 22.0, 24.0])
    numero_restricciones = len(restriccion_peso_max)

    restriccion_peso_por_item = np.array([[8.0, 12.0, 13.0, 64.0, 22.0, 41.0],
                                          [8.0, 12.0, 13.0, 75.0, 22.0, 41.0],
                                          [3.0, 6.0, 4.0, 18.0, 6.0, 4.0],
                                          [5.0, 10.0, 8.0, 32.0, 6.0, 12.0],
                                          [5.0, 13.0, 8.0, 42.0, 6.0, 20.0],
                                          [5.0, 13.0, 8.0, 48.0, 6.0, 20.0],
                                          [0.0, 0.0, 0.0, 0.0, 8.0, 0.0],
                                          [3.0, 0.0, 4.0, 0.0, 8.0, 0.0],
                                          [3.0, 2.0, 4.0, 0.0, 8.0, 4.0],
                                          [3.0, 2.0, 4.0, 8.0, 8.0, 4.0]])

    factor_penalidad = 5000.0

    vector_recompensa = np.zeros(particulas)
    matriz_pesos = np.zeros((particulas, numero_restricciones))
    matrix_violacion_restricciones = np.zeros(
        (particulas, numero_restricciones))
    vector_penalidad = np.zeros(particulas)
    recompensa_penalizada = np.zeros(particulas)

    for i in range(particulas):
        vector_recompensa[i] = 0.0
        for j in range(dimensiones):
            
            vector_recompensa[i] = reconpensa_por_item[j] * posibles_soluciones[i, j] + vector_recompensa[i]  # Funcion objetivo

        for k in range(numero_restricciones):
            matriz_pesos[i, k] = 0.0
            for j in range(dimensiones):
                # Restricciones
                matriz_pesos[i, k] = restriccion_peso_por_item[k,j] * posibles_soluciones[i, j] + matriz_pesos[i, k]
            # Violaciones
            matrix_violacion_restricciones[i,k] = matriz_pesos[i, k] - restriccion_peso_max[k]

        vector_penalidad[i] = 0.0

        for k in range(numero_restricciones):
            if (matrix_violacion_restricciones[i, k] <= 0.0):
                matrix_violacion_restricciones[i, k] = 0.0
            # Penalidad
            vector_penalidad[i] = matrix_violacion_restricciones[i,
                                                                 k] + vector_penalidad[i]
            # Func.Obj.con penalizacion
        recompensa_penalizada[i] = vector_recompensa[i] - \
            factor_penalidad * vector_penalidad[i]

    return -recompensa_penalizada




def test_mochila():
    # 6 items, 3 particulas,
    posibles_soluciones_3_particulas = np.array([[1, 1, 1, 1, 1, 1],  # todos en la mochila
                                                 # ninguno en la mochila
                                                 [0, 0, 0, 0, 0, 0],
                                                 [1, 0, 1, 0, 1, 0],
                                                 [0, 1, 1, 0, 0, 1]])  # algunos

    result = mochila(posibles_soluciones_3_particulas)
    print("result vector")
    print(result)


def discrete_optimization():
    print("Ejemplo optimizacion discreta")
    numero_items = 6
    numero_particulas = 100

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': numero_particulas, 'p': 1}

    # Call instance of PSO
    optimizer = ps.discrete.binary.BinaryPSO(n_particles=numero_particulas, dimensions=numero_items, options=options,
                                             init_pos=None,
                                             velocity_clamp=None, vh_strategy='unmodified', ftol=1.0)

    # Perform optimization
    return optimizer.optimize(mochila, iters=100,)




if __name__ == "__main__":

    print("Running Benchmark... ")
    args = read_args()

    args.fn = mochila
    
    args.discrete = True
    
    
    
    args.n = 100
    args.d = 6
    start = time.time()
    r1 = discrete_optimization()
    t1 = time.time() - start
  
    start = time.time()
    r2 = PSO.raw_implementation(args)
    t2 = time.time() - start
    
    
    
    print("*"*50)
    print("PySwarms time : ", t1)
    print("PySwarms sol : ", r1[0])
    print("PySwarms pos : ", r1[1])
    print("*"*50)
    print("UTN time : ", t2)
    print("UTN sol : ", r2[1])
    print("UTN pos : ",(r2[2]))

