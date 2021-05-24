#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sys import argv
import matplotlib.pyplot as plt
from mimsbi.generate import ConditionalGenerator
from utils import return_pars

#input model
model=argv[1]

# load pars and generate
n_par,prior_sample,boundaries,default_measurement_time,prior,simulator=return_pars(model)
x = ConditionalGenerator(int(1e5),prior_sample,simulator)

#save 
folder='~/minimalistic/results/data/'
filename = folder+'data_prior_sample_{}'.format(model).replace('.','_') + '.csv.gz'
df = pd.DataFrame(x)
df.to_csv(filename,index=False)