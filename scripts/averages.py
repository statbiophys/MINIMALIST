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

averages={}
for model in ['bd', 'ou','sir','lorentz']:
    folder='results/'
    code='data_{}'.format(model).replace('.','_') 
    data=pd.read_csv(folder+'data/'+code + '.csv.gz')
    n_par,prior_sample,boundaries,default_measurement_time,prior,simulator= return_pars(model)

    all_xs=data.values[:,n_par:]
    mean=all_xs.mean(axis=0)
    std=all_xs.std(axis=0)
    
    averages['mean_'+model]=mean
    averages['std_'+model]=std
    averages={'mean':mean,'std':std}
    np.save(folder+'data/averages_'+model+'.npy', averages)
    
#np.save(folder+'data/averages.npy', averages)