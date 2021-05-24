#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sys import argv
import matplotlib.pyplot as plt
from mimsbi.generate import Generator,ConditionalGenerator
from utils import return_pars

#input model
model=argv[1]

# load pars and generate
n_par,prior_sample,boundaries,default_measurement_time,prior,simulator=return_pars(model)
theta,x = Generator(int(2e7),prior,simulator)

#save 
folder='results/data/'
filename = folder+'data_{}'.format(model).replace('.','_') + '.csv.gz'
df = pd.DataFrame(np.concatenate([theta,x],axis=1))
df.to_csv(filename,index=False)

#save mean and averages
mean=x.mean(axis=0)
std=x.std(axis=0)
averages={'mean':mean,'std':std}
np.save(folder+'averages_'+model+'.npy', averages)

# Conditional generation
x = ConditionalGenerator(int(1e5),prior_sample,simulator)
filename = folder+'data_prior_sample_{}'.format(model).replace('.','_') + '.csv.gz'
df = pd.DataFrame(x)
df.to_csv(filename,index=False)