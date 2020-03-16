from common import *
import psHandler as psh
import pso as PSO
# import psoJAX as PSOjax
import knapspack as knapspack
from params import params


if __name__ == "__main__":

    print("Running Benchmark... ")

    knp = knapspack.Knapspack()

    params.fn = knp.mochila

    start = time.time()
    t1, c1, p1 = psh.discrete(params)
    tt1 = time.time() - start

    start = time.time()
    t2, c2, p2 = PSO.discrete(params)
    tt2 = time.time() - start
    
    start = time.time()
    t3, c3, p3 = PSO.discrete(params)
    tt3 = time.time() - start

    """
    start = time.time()
    t2, c2, p2 = PSOps.using_bounds(args)
    tt2 = time.time() - start

    start = time.time()
    t3, c3, p3 = PSOjax.raw_implementation(args)
    tt3 = time.time() - start

    start = time.time()
    t4, c4, p4 = PSO.raw_implementation(args)
    tt4 = time.time() - start
     """

    print("Results ......")
    print("PySwarms time: \t", t1, "\t", tt1)
    print("c: \t", c1)
    print("p: \t", p1)
    print("vio: \t", knp.check_vio(p1))


    print("PSO time: \t", t2, "\t", tt2)
    print("c: \t", c2)
    print("p: \t", p2)
    print("violations: \t", knp.check_vio(p2))
    
    print("PSOjax time: \t", t3, "\t", tt3)
    print("c: \t", c3)
    print("p: \t", p3)
    print("violations: \t", knp.check_vio(p3))

    # print ("Serie time: \t", t4,"\t",tt4)
