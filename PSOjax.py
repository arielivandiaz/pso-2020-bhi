# import numpy as np
import jax.numpy as np
from jax import random
from jax import grad, jit
import time


@jit
def calculate_velocity(w, particles_velocity, c1, c2, r1, r2, best_particle_position, particles_position,
                       best_global_position):
    inertia = w * particles_velocity
    best_particle_pos_component = r1 * (best_particle_position - particles_position)
    best_global_pos_component = r2 * (best_global_position - particles_position)

    new_velocity = inertia + c1 * best_particle_pos_component + c2 * best_global_pos_component
    return new_velocity


@jit
def calculate_position(particles_position, particles_velocity):
    return particles_position + particles_velocity


@jit
def calculate_best_position(f, best_particle_cost, particles_position, best_particle_position,particles,dimensions):
    bests = np.less(f, best_particle_cost)
    best_particle_cost = np.where(bests, f, best_particle_cost)
    reshape = np.reshape(bests, np.array([particles, 1]))
    bests_reshape = np.broadcast_to(reshape, np.array([particles, dimensions]))
    pos = np.where(bests_reshape, particles_position, best_particle_position)
    return pos


def runPSO(user_options,algorith_options):
    key = random.PRNGKey(0)

    box = algorith_options['box']
    # For each particle, initialize position and velocity
    particles_position = random.uniform(key, (algorith_options['particles'], algorith_options['dimensions']), None, -box, box)
    particles_velocity = random.uniform(key, (algorith_options['particles'], algorith_options['dimensions']), None, -box, box)

    k = 0
    particles = algorith_options['particles']
    dimensions = algorith_options['dimensions']
    best_global = None  # Best swarm cost
    best_global_position = np.empty((particles, dimensions))  # Best swarm position
    best_particle_position = particles_position
    best_particle_cost = algorith_options['objective'](best_particle_position) #sphere(best_particle_position)


    while k < algorith_options['iterations']:
        f = algorith_options['objective'](best_particle_position) #sphere(particles_position)
        best_index = np.argmin(f)
        best_value = f[best_index]

        best_particle_position = calculate_best_position(f, best_particle_cost, particles_position,
                                                         best_particle_position,particles,dimensions)

        if best_global is None or best_value < best_global:
            # Update best swarm cost and position
            best_global = best_value
            best_global_position = particles_position[best_index]

        # Generate r1 and r2 for each particle and iteration.
        r1 = random.uniform(key, (algorith_options['particles'], algorith_options['dimensions']), None, 0, 1)
        r2 = random.uniform(key, (algorith_options['particles'], algorith_options['dimensions']), None, 0, 1)

        # Update velocity
        particles_velocity = calculate_velocity(user_options['w'], particles_velocity, user_options['c1'], user_options['c2'], r1, r2, best_particle_position,
                                                particles_position, best_global_position)

        # Update position
        particles_position = calculate_position(particles_position, particles_velocity)

        k += 1

    return best_global, best_global_position


def raw_implementation(args):
    user_options = {'c1':args.c1, 'c2':args.c2, 'w': args.w}
    algorith_options = {'particles': args.n, 'dimensions': args.d, 'iterations': args.i, 'objective': args.fn, 'box' : args.box}

    start = time.time()
    best_global, best_global_position = runPSO(user_options,algorith_options)     
    return time.time() - start,  best_global, best_global_position 





