# import numpy as np
import numpy as np
import time
import random
from common import *

#################################################################################################
def calculate_velocity(w, particles_velocity, c1, c2, r1, r2, best_particle_position, particles_position,
                       best_global_position):
    inertia = w * particles_velocity
    best_particle_pos_component = r1 * \
        (best_particle_position - particles_position)
    best_global_pos_component = r2 * \
        (best_global_position - particles_position)

    new_velocity = inertia + c1 * best_particle_pos_component + \
        c2 * best_global_pos_component
    return new_velocity

#################################################################################################
def calculate_position(particles_position, particles_velocity):
    return particles_position + particles_velocity

#################################################################################################
def calculate_best_position(f, best_particle_cost, particles_position, best_particle_position, particles, dimensions):
    bests = np.less(f, best_particle_cost)
    best_particle_cost = np.where(bests, f, best_particle_cost)
    reshape = np.reshape(bests, np.array([particles, 1]))
    bests_reshape = np.broadcast_to(reshape, np.array([particles, dimensions]))
    pos = np.where(bests_reshape, particles_position, best_particle_position)
    return pos

#################################################################################################
def runPSO(params):

    box = algorith_options['box']
    # For each particle, initialize position and velocity
    particles_position = np.random.uniform(
        -box, box, (params.n, params.d))
    particles_velocity = np.random.uniform(
        -box, box, (params.n, params.d))

    k = 0
    particles = params.n
    dimensions = params.d
    best_global = None  # Best swarm cost
    best_global_position = np.empty(
        (particles, dimensions))  # Best swarm position
    best_particle_position = particles_position
    best_particle_cost = params.fn(
        best_particle_position)  # sphere(best_particle_position)

    while k < params.i:
        f = params.fn(
            best_particle_position)  # sphere(particles_position)
        best_index = np.argmin(f)
        best_value = f[best_index]

        best_particle_position = calculate_best_position(f, best_particle_cost, particles_position,
                                                         best_particle_position, particles, dimensions)

        if best_global is None or best_value < best_global:
            # Update best swarm cost and position
            best_global = best_value
            best_global_position = particles_position[best_index]

        # Generate r1 and r2 for each particle and iteration.
        r1 = 0.52 #np.random.uniform(0, 1, (params.n, params.d))
        r2 = 0.48 #np.random.uniform(0, 1, (params.n, params.d))

        # Update velocity
        particles_velocity = calculate_velocity(user_options['w'], particles_velocity, user_options['c1'], user_options['c2'], r1, r2, best_particle_position,
                                                particles_position, best_global_position)

        # Update position
        particles_position = calculate_position(
            particles_position, particles_velocity)

        k += 1

    return best_global, best_global_position

#################################################################################################
def runDiscretePSO(params):



    
    # For each particle, initialize position and velocity
    particles_position = np.random.uniform(
        -1, 1,(params.n, params.d))
    particles_velocity = np.random.uniform(
        -1, 1, (params.n, params.d))
    
    particles_position = toDiscrete(activation(particles_position))
  
  
    k = 0
    particles = params.n
    dimensions = params.d
    best_global = None  # Best swarm cost
    best_global_position = np.empty(
        (particles, dimensions))  # Best swarm position
    best_particle_position = particles_position
    best_particle_cost = params.fn(best_particle_position)  # sphere(best_particle_position)

    #while k < params.i:
    for k in range(params.i):
        f = params.fn(best_particle_position)  # sphere(particles_position)
        best_index = np.argmin(f)        
        best_value = f[best_index]
        
        best_particle_position = calculate_best_position(f, best_particle_cost, particles_position,
                                                         best_particle_position, particles, dimensions)
 
        if best_global is None or best_value < best_global:
            # Update best swarm cost and position
            best_global = best_value
            best_global_position = particles_position[best_index]
            
        

        r1 = np.random.uniform(0, 1, (params.n, params.d))
        r2 = np.random.uniform(0, 1, (params.n, params.d))

 
        particles_velocity = calculate_velocity(params.options['w'], particles_velocity, params.options['c1'], params.options['c2'], r1, r2, best_particle_position,
                                                particles_position, best_global_position)
  

        # Update position
        particles_position =  toDiscrete(activation((calculate_position(particles_position, particles_velocity))))

        k += 1

    return best_global, best_global_position




#################################################################################################
def discrete(args):
    
    start = time.time()
    best_global, best_global_position = runDiscretePSO(args)

    return time.time() - start,  best_global, best_global_position
    
#################################################################################################
def raw_implementation(args):

    user_options = {'c1': args.options['c1'], 'c2': args.options['c2'], 'w': args.options['w']}
    algorith_options = {'particles': args.n, 'dimensions': args.d,
                        'iterations': args.i, 'objective': args.fn, 'box': 1}

    start = time.time()
    if (args.discrete):
        print("Discrete PSO")
        best_global, best_global_position = runDiscretePSO(args)
    else:
        best_global, best_global_position = runPSO(
            user_options, algorith_options)

    return time.time() - start,  best_global, best_global_position
