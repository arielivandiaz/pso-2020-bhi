from common import *
import psHandler as psh
import pso as PSO
# import psoJAX as PSOjax
import knapspack as knapspack
from params import params


VERBOSE = 0


def run(solver=0, file=0):

    knp = knapspack.Knapspack()
    params.fn = knp.mochila
    
    

    if (solver==1 or solver == 0):
        start = time.time()
        t1, c1, p1 = psh.discrete(params)
        tt1 = time.time() - start
        if (VERBOSE):
            print("Results ......")
            print("PySwarms time: \t", t1, "\t", tt1)
            print("c: \t", c1)
            print("p: \t", p1)
            print("vio: \t", knp.check_vio(p1))
        print( t1,'\t', tt1,'\t',c1,'\t',knp.check_vio(p1))
        if(file):
            file.write( "%lf \t %lf \t %lf \t %d \n\r " % (t1, tt1,c1,knp.check_vio(p1)))
    
    if (solver==2 or solver == 0):
        start = time.time()
        t2, c2, p2 = PSO.discrete(params)
        tt2 = time.time() - start
        if (VERBOSE):
            print("PSO time: \t", t2, "\t", tt2)
            print("c: \t", c2)
            print("p: \t", p2)
            print("violations: \t", knp.check_vio(p2))
        print( t2,'\t', tt2,'\t',c2,'\t',knp.check_vio(p2))
        if(file):
            file.write( "%lf \t %lf \t %lf \t %d \n\r " % (t2, tt2,c2,knp.check_vio(p2)))


    if (solver==3 or solver == 0):
        start = time.time()
        t3, c3, p3 = PSO.discrete(params)
        tt3 = time.time() - start
        if (VERBOSE):
            print("PSOjax time: \t", t3, "\t", tt3)
            print("c: \t", c3)
            print("p: \t", p3)
            print("violations: \t", knp.check_vio(p3))
        print( t3,'\t', tt3,'\t',c3,'\t',knp.check_vio(p3))
        if(file):
            file.write( "%lf \t %lf \t %lf \t %d \n\r " % (t3, tt3,c3,knp.check_vio(p3)))


def test(param, pfrom, pto):
    
    f= open("benchmark.txt","w+")
    
    run(file=f)
    
    f.close() 


if __name__ == "__main__":

    print("Running Benchmark... ")

    #run()
    
    
    test(1,2,3)
    """

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
    
    """
