import subprocess
from sys import argv
import os
import pandas as pd
import numpy as np
model=str(argv[1])

regs=[1e-4,1e-5,1e-6]
batch_sizes=[1e3,1e4]
learning_rates=[1e-2,1e-3,1e-4]

for batch_size in batch_sizes:
    for lr in learning_rates:
        to_run=[]            
        for reg in regs:
            code=model+' '+str(reg)+' '+str(int(batch_size))+' '+str(lr)
            to_run.append('python scripts/hyperpars.py '+code)                                                       
        subprocess.run('& '.join(to_run), shell=True)
        
# save the results        
data = os.listdir('results/hyperparameters/')
data=[d for d in data if model in d]

dfs=[]
for d in data:
    dfs.append(pd.read_csv('results/hyperparameters/'+d))
final=pd.concat(dfs)

dictionary={}
for objective in ['MINE','BCE','fMINE']:

    df1=final.loc[final.objective==objective].groupby(['reg','batch_size','learning_rate']).MI.mean().reset_index()
    df2=final.loc[final.objective==objective].groupby(['reg','batch_size','learning_rate']).MI.std().reset_index()
    
    merged=df1.merge(df2,on=['reg','batch_size','learning_rate'])
    merged['diff']=merged.MI_x-merged.MI_y
    merged=merged.sort_values('diff',ascending=False)
    reg,batch_size,learning_rate,_,_,_=merged.iloc[0].values
    dictionary[objective]=[reg,int(batch_size),learning_rate]
    
np.save('results/data/hyperpars_'+model+'.npy',dictionary)