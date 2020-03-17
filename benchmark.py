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
            file.write( "%lf \t %lf \t %lf \t %d \n " % (t1, tt1,c1,knp.check_vio(p1)))
    
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
            file.write( "%lf \t %lf \t %lf \t %d \n " % (t2, tt2,c2,knp.check_vio(p2)))


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
            file.write( "%lf \t %lf \t %lf \t %d \n " % (t3, tt3,c3,knp.check_vio(p3)))


def test(solver,param, pfrom, step,pto):
    
    f= open("benchmark.txt","w+")
    if(solver == 'pyswarms'):
        nsolver=1
    elif (solver == 'pso'):
        nsolver=2
    elif (solver == 'psojax'):
        nsolver=3
    else:
        nsolver=0
    i = pfrom
    while (i<=pto):
        if (param == 'items'):
            params.items=i
            params.d=i
        elif (param == 'iters'):
            params.i=i
        elif (param == 'rest'):
            params.rest = i
        elif(param == 'n'):
            params.n=i        
        else:
            print("ERROR: test param not found")
        run(nsolver,f)
        i+=step
    
    f.close() 

def print_header(msg):

    f= open("benchmark.txt","w+")
    f.write(msg)
    f.write("iters : %d\n" % params.i)
    f.write("items : %d\n" % params.items)
    f.write("restricciones : %d\n" % params.rest)
    f.write("particulas : %d\n" % params.n)

    f.close() 

def reset_params():

    params.i = 100 #iters
    params.items = 50
    params.rest = 4 #restrictions
    params.n = 50    
    params.d = params.items


if __name__ == "__main__":

    print("Running Benchmark... ")

    #Run pyswars, pso & psojax:
    #run() #check verbose ==1

    print_header("items de 25 a 150 PS")
    test('pyswarms','items',25,5,150)
    print_header("items de 25 a 150 PSO")
    test('pso','items',25,5,150)

    reset_params()

    print_header("iters de 100 a 2000 PS")
    test('pyswarms','iters',100,100,2000)
    pprint_header("iters de 100 a 2000 PSO")
    test('pso','iters',100,100,2000)

    reset_params()

    print_header("rest de 0 a 20 PS")
    test('pyswarms','rest',0,1,20)
    pprint_header("rest de 0 a 20 PSO")
    test('pso','rest',0,1,20)

    reset_params()

    print_header("particles de 5 a 150 PS")
    test('pyswarms','n',5,5,150)
    pprint_header("particles de 5 a 150 PSO")
    test('pso','n',5,5,150)
   
