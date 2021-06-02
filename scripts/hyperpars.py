#!/usr/bin/env python
# coding: utf-8

import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from mimsbi.divergences import MutualInformation
import matplotlib.pyplot as plt
import pandas as pd
from sys import argv
from tqdm import tqdm
from utils import *

n_simulations=int(2e5)
objectives=['MINE','BCE','fMINE']
reps=4

model=str(argv[1])
reg=float(argv[2])
batch_size=int(argv[3])
learning_rate=float(argv[4])
seeds=(np.arange(reps)*20).astype(np.int)
n_par,prior_sample,boundaries,default_measurement_time,prior,simulator= return_pars(model)

# read data
folder='results/'
code='data_{}'.format(model)
data=pd.read_csv(folder+'data/'+code + '.csv.gz')
all_xs=data.values[:,n_par:]
mean=all_xs.mean(axis=0)
std=all_xs.std(axis=0)
final_dfs=[]

results=[]
for objective in objectives:
    for rep in tqdm(range(reps)):
        code='{}_reg{}_batch{}_lr{}_obj{}_rep{}'.format(model,reg,batch_size,learning_rate,objective,rep).replace('.','_')
        subsampled=data.sample(n=n_simulations,random_state=seeds[rep]) # different subsample for test
        theta,x_=subsampled.values[:,:n_par],subsampled.values[:,n_par:]
        subsample_test=data.loc[data.index.difference(subsampled.index), ]
        theta_test,x_test_=subsample_test.values[:,:n_par],subsample_test.values[:,n_par:]

        # normalise data
        x=(x_-mean)/std
        x_test=(x_test_-mean)/std

        estimator=MutualInformation(values1=theta,values2=x,objective=objective,seed=seeds[rep],
                                    l2_reg=reg,validation_split=0.5,lr=learning_rate)
        estimator.fit(epochs=2000,batch_size=batch_size,seed=seeds[rep],weights_name='results/weights/'+code+'.h5')

        #evaluate
        MI, BCE= estimator.evaluate(theta_test,x_test)    
        results.append([rep,objective,reg,batch_size,MI,BCE,learning_rate])

df=pd.DataFrame(results,columns=['rep','objective','reg','batch_size','MI','BCE','learning_rate'])
code='{}_reg{}_batch{}_lr{}'.format(model,reg,batch_size,learning_rate).replace('.','_')
df.to_csv(folder+'hyperparameters/'+code+'.csv',index=False)