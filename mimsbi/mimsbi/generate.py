from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
from numpy.random import seed
from time import time

def run_simulation(x):
    seed(x[0])
    prior=x[2]
    simulator=x[1]
    true_params=prior.sample()[0]
    return [true_params,simulator.simulate(true_params)]

def Generator(n,prior,simulator,verbose=True):

    simulators=[simulator for _ in range(n)]
    priors=[prior for _ in range(n)]
    seeds=(((np.arange(n)+1)*time())*100%2**32).astype(np.int)
    
    with Pool(cpu_count()-2) as p:
        ret_list = list(tqdm(p.imap(run_simulation,zip(seeds,simulators,priors)),total=n,disable=not verbose))
        
    length=len(ret_list)
    shape0=len(ret_list[0][0])
    shape1=len(ret_list[0][1])
    inputs_,outputs_=np.array(list(np.array(ret_list)[:,0])).reshape(length,shape0),np.array(list(np.array(ret_list)[:,1])).reshape(length,shape1)
    return inputs_,outputs_

def run_simulation_fixed_parameter(x):
    seed(x[0])
    return x[1].simulate(x[2])

def ConditionalGenerator(n,prior,simulator):
    simulators=[simulator for _ in range(n)]
    priors=[prior for _ in range(n)]
    seeds=(((np.arange(n)+1)*time())*100%2**32).astype(np.int)

    with Pool(cpu_count()-2) as p:
        ret_list = list(tqdm(p.imap(run_simulation_fixed_parameter,zip(seeds,simulators,priors)),total=n))
    return ret_list