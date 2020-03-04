import argparse
import time
import numpy as np 
import random

def activation(x, function=1):

    if (function == 1):  # Sigmoid        
        return 1/(1+np.exp(-x))
    elif (function == 2):  # Softmax
        expo = np.exp(x)
        expo_sum = np.sum(np.exp(x))
        return expo/expo_sum
    elif (function == 3):  # ReLu
        return np.maximum(0, x)


def getShapeasArray(x):

    y = x.shape
    shape = []
    for i in y:
        shape.append(i)
    return shape


def toDiscrete(x):

    
    return np.where(x < 0.5, 0, 1)
    #return np.where(x < random.random(), 0, 1) #no funciona
    


def test_toDiscrete():

    x = np.random.rand(1, 20)

    print(x) 
    print(toDiscrete(x))
    print(toDiscrete(activation(x)))


def read_args():

    parser = argparse.ArgumentParser(description='Simple PSO Benchmark.')

    # w
    parser.add_argument('-w',  type=float, default=0.9,
                        help="Constant inertia weight")
    # c1
    parser.add_argument('-c1',  type=float, default=0.5,
                        help="Cognitive term weight")
    # c2
    parser.add_argument('-c2',  type=float, default=0.3,
                        help="Group term weight")
    # n
    parser.add_argument('-n',  type=int, default=100, help="Swarm size")
    # d
    parser.add_argument('-d',  type=int, default=1000, help="Dimensions")
    # i
    parser.add_argument('-i',  type=int, default=100, help="Iterations")
    # box
    parser.add_argument('-box',  type=float, default=1, help="Box limit")
    # x0
    parser.add_argument('-x0',  type=float, default=5.0,
                        help="Swarm initial position")
    # fn
    parser.add_argument('-fn',  type=int, default=1, help="Cost function id")
    # loops
    parser.add_argument('-loops',  type=int, default=15,
                        help="Loops (Benchmark)")
    # Verbose
    parser.add_argument('--discrete',  action='store_true',
                        help="Discrete optimitation")
    # Verbose
    parser.add_argument('--verbose',  action='store_true',
                        help="Enable partial prints")
    # File
    parser.add_argument('--file',  action='store_true',
                        help="Enable file output prints")
    # Benchmark
    # bench_flag
    parser.add_argument('-bench_flag',  type=int,
                        default=1, help="Value to test")
    # bench_step
    parser.add_argument('-bench_step',  type=float,
                        default=20, help="Benchmark step")
    # bench_start
    parser.add_argument('-bench_start',  type=float,
                        default=10.0, help="Benchmark start value")
    # bench_end
    parser.add_argument('-bench_end',  type=float, default=105.0,
                        help="Benchmark last value")
    # bench_name
    parser.add_argument('-bench_name',  type=str,
                        help="Benchmark name")

    
    return parser.parse_args()

