#!/usr/bin/env python
# coding: utf-8

import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from mimsbi.divergences import MutualInformation
from mimsbi.generate import ConditionalGenerator
import matplotlib.pyplot as plt
import pandas as pd
from sys import argv
from tqdm import tqdm
from utils import *
plt.style.use('seaborn-white')


# inputs
model=argv[1]
n_simulations=int(argv[2])
objective=str(argv[3])
reps=10
val_split=0.5
n_sims=n_simulations


seeds=np.arange(reps).astype(np.int)
n_par,prior_sample,boundaries,default_measurement_time,prior,simulator= return_pars(model)

# read data
folder='~/minimalistic/results/'
code='data_{}'.format(model).replace('.','_') 
data=pd.read_csv(folder+'data/'+code + '.csv.gz')

regularization,batch_size,learning_rate=return_hyperpars(objective,model)

if n_simulations> 3e6: 
    reps=1
    n_sims=1e7
    
all_xs=data.values[:,n_par:]
mean=all_xs.mean(axis=0)
std=all_xs.std(axis=0)
final_dfs=[]

for rep in tqdm(range(reps)):
    code='{}_sim{}_obj{}_rep{}'.format(model,int(np.log10(n_sims)),objective,rep).replace('.','_')
    subsampled=data.sample(n=n_simulations,random_state=seeds[rep]) # different subsample for test
    theta,x_=subsampled.values[:,:n_par],subsampled.values[:,n_par:]
    subsample_test=data.loc[data.index.difference(subsampled.index), ]
    theta_test,x_test_=subsample_test.values[:,:n_par],subsample_test.values[:,n_par:]

    # normalise data
    x=(x_-mean)/std
    x_test=(x_test_-mean)/std
    
    estimator=MutualInformation(values1=theta,values2=x,objective=objective,seed=seeds[rep],l2_reg=regularization,validation_split=val_split,lr=learning_rate)
    estimator.fit(epochs=5000,batch_size=batch_size,seed=seeds[rep],weights_name='results/weights/'+code+'_mi.h5')
    
    #evaluate
    MI, BCE= estimator.evaluate(theta_test,x_test)    
    estimator.mi_value = MI
    estimator.bce_value = BCE

    #rejection sampling
    prior_samples=np.array([prior_sample for i in range(len(x_test))])
    ratios=estimator.log_ratio(prior_samples,x_test)
    accepted_samples=np.random.uniform(size=len(ratios))<sigmoid(ratios)
    print(np.sum(accepted_samples))
    
    accepted_test=x_test[accepted_samples][:int(5e4)] # take only the first 50000 ones.
    estimator.save_model('results/estimators/'+code)
    pd.DataFrame(accepted_test).to_csv('results/estimators/'+code+'/accepted_samples.csv.gz',compression='gzip',index=False)
    plot_training(estimator,savename='results/estimators/'+code+'/_training.png')