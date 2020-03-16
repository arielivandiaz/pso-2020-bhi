import numpy as np
import time

# Import PySwarms
import pyswarms as ps
from pyswarms.single.global_best import GlobalBestPSO



def basic_optimization(args):
    dimensions = args.d
    particles = args.n
    options = {'c1': args.c1, 'c2': args.c2, 'w': args.w}

    optimizer = ps.single.GlobalBestPSO(
        n_particles=particles, dimensions=dimensions, options=options)

    start = time.time()
    cost, pos = optimizer.optimize(args.fn, iters=args.i)
    
    return time.time() - start, cost, pos
    


def using_bounds(args):
    dimensions = args.d
    particles = args.n

    max_bound = args.box * np.ones(dimensions)
    min_bound = - max_bound
    bounds = (min_bound, max_bound)

    options = {'c1': args.c1, 'c2': args.c2, 'w': args.w}

    optimizer = ps.single.GlobalBestPSO(
        n_particles=particles, dimensions=dimensions, options=options, bounds=bounds)

    start = time.time()
    cost, pos = optimizer.optimize(args.fn, iters=args.i)
    
    return time.time() - start, cost, pos


def discrete (args):
    N = args.items
    P = args.n

    options = {'c1': args.options['c1'], 'c2': args.options['c2'],
               'w': args.options['w'], 'k': args.options['k'], 'p': args.options['p']}

    optimizer = ps.discrete.binary.BinaryPSO(n_particles=P, dimensions=N, options=options, init_pos=None,
                                             velocity_clamp=None, vh_strategy='unmodified', ftol=0)

    start = time.time()    
    [solFO, solX] = optimizer.optimize(args.fn, iters=args.i)
    
    return time.time() - start, solFO, solX
    
    