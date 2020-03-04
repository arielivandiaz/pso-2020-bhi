from common import *
import PSOpyswarms as PSOps
import PSO
import PSOjax 
import functions as fx
import functions as fxJAX


if __name__ == "__main__":

    print("Running Benchmark... ")
    args = read_args()

    if args.fn == 1:
        args.fn = fx.sphere
    elif args.fn == 2:
        args.fn = fxJAX.sphere
    else:
        print("ERROR : FUNCTION NOT FOUND")

    """
    start = time.time()
    t1, c1, p1 = PSOps.basic_optimization(args)
    tt1 = time.time() - start

    start = time.time()
    t2, c2, p2 = PSOps.using_bounds(args)
    tt2 = time.time() - start

    start = time.time()
    t3, c3, p3 = PSOjax.raw_implementation(args)
    tt3 = time.time() - start
        """
    start = time.time()
    t4, c4, p4 = PSO.raw_implementation(args)
    tt4 = time.time() - start
        



    print ("Results ......")
    #print ("Basic time: \t", t1,"\t",tt1)
    #print ("Bounds time: \t", t2,"\t",tt2)
    #print ("JAX time: \t", t3,"\t",tt3)
    print ("Serie time: \t", t4,"\t",tt4)
    
    