from scripts.utils import *
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mimsbi.models.ou import Posterior
import matplotlib as mlp
plt.style.use('seaborn-white')

mlp.rcParams.update({'font.size': 14})
models=['ou','bd','sir']
n_obs={'sir':2,'ou':5,'bd':5}
n=100
x_labels1=[r'$\sigma$',r'$\alpha$',r'$\beta$']
x_labels2=[r'$\mu$',r'$\beta$',r'$\gamma$']
model_names=['Ornstein–Uhlenbeck','Birth-Death','SIR','Lorenz attractor']
letters=list('ABCDEFGHIJKLMNO')
l=0

fig=plt.figure(figsize=(12,6),dpi=200)
for j,model in enumerate(models):
    np.random.seed(123)
    objectives=['MINE','fMINE','BCE']
    estimators=[]
    
    for obj in objectives: estimators.append(MutualInformation(load_dir='results/estimators/'+model+'_sim7_obj'+obj+'_rep0'))
        
    averages=np.load('results/data/averages_'+model+'.npy',allow_pickle=True).item()
    mean=averages['mean']
    std=averages['std']
    observations=pd.read_csv('results/data/data_prior_sample_'+model+'.csv.gz').values
    obs=(observations-mean)/std
    n_par,prior_sample,boundaries,default_measurement_time,prior,simulator= return_pars(model)
       
    matrices=[]
    o=obs[np.random.choice(np.arange(len(obs)),size=n_obs[model])]
    for i,estimator in enumerate(tqdm(estimators)):
        m,spacex,spacey=scan2d(o,estimator,boundaryx=boundaries[0],boundaryy=boundaries[1])
        matrices.append(m)
        
    if model=='ou':
        simulator.means=mean
        simulator.stds=std
        posterior=Posterior(simulator)
        objectives=['MINE','fMINE','BCE','true']
        m,_,_=scan2d(o,posterior,boundaryx=[boundaries[0][0]+1e-5,boundaries[0][1]-1e-5],boundaryy=boundaries[1])
        matrices.append(m)
    objectives_names=['MINE','FDIV','BCE','true']

    locationy=np.arange(n)[spacey>prior_sample[1]].min()
    locationx=np.arange(n)[spacex>prior_sample[0]].min()
    indexes_x=['%0.f'%f for f in spacex]
    indexes_y=['%0.f'%f for f in spacey]
    ticks=np.linspace(0,n-1,2).astype(np.int)
    if model=='ou': 
        ax=plt.subplot(2,4,5+j)
        ax.annotate(letters[l+1], xy=(-0.05, 1.15), xycoords='axes fraction', textcoords='offset points', fontsize=15,xytext=(0, -5), weight='bold', ha='right',  va='top')

    else: 
        ax=plt.subplot(2,4,1+j)
        plt.title(model_names[j],y=1.08)
        ax.annotate(letters[l], xy=(-0.05, 1.15), xycoords='axes fraction', textcoords='offset points', fontsize=15,xytext=(0, -5), weight='bold', ha='right',  va='top')


    #ax.annotate(letters[l], xy=(-0.05, 1.15), xycoords='axes fraction', textcoords='offset points', fontsize=15,xytext=(0, -5), weight='bold', ha='right',  va='top')
    l+=1
    plt.locator_params(axis='y', nbins=4)

    for i,m in enumerate(matrices):
        plt.plot(m.sum(axis=1),label=objectives_names[i])
        plt.axvline(locationx,c='k')
        plt.xticks(ticks,[indexes_x[int(i)] for i in ticks])
    plt.xlabel(x_labels1[j])
    #plt.legend(frameon=False,handlelength=1)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0),useMathText=True)
    if model=='ou': 
        plt.ylabel('Posterior Density')
        ax=plt.subplot(2,4,1+j)
        plt.title(model_names[j],y=1.08)
        ax.annotate("true value",xy=(locationy,0.01),xytext=(locationy-70, 0.015),xycoords='data', textcoords='data',
            arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),
            )
        ax.annotate(letters[l-1], xy=(-0.05, 1.15), xycoords='axes fraction', textcoords='offset points', fontsize=15,xytext=(0, -5), weight='bold', ha='right',  va='top')

    else: 
        ax=plt.subplot(2,4,5+j)
        ax.annotate(letters[l], xy=(-0.05, 1.15), xycoords='axes fraction', textcoords='offset points', fontsize=15,xytext=(0, -5), weight='bold', ha='right',  va='top')

        
    l+=1
    plt.locator_params(axis='y', nbins=4)

    for i,m in enumerate(matrices):
        plt.plot(m.sum(axis=0),label=objectives_names[i])
        plt.axvline(locationy,c='k')
        plt.xticks(ticks,[indexes_y[int(i)] for i in ticks])
    if model=='ou': plt.legend(frameon=False,handlelength=1)
    plt.xlabel(x_labels2[j])
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0),useMathText=True)
    if model=='ou': plt.ylabel('Posterior Density')



# 1D 
        
model='lorentz'

estimators=[]
for obj in objectives:
    estimator=MutualInformation(load_dir='results/estimators/'+model+'_sim7_obj'+obj+'_rep0')
    estimators.append(estimator)

averages=np.load('results/data/averages_'+model+'.npy',allow_pickle=True).item()
mean=averages['mean']
std=averages['std']
observations=pd.read_csv('results/data/data_prior_sample_'+model+'.csv.gz').values
obs=(observations-mean)/std
n_par,prior_sample,boundaries,default_measurement_time,prior,simulator= return_pars(model)

matrices=[]
o=obs[np.random.choice(np.arange(len(obs)),size=5)]
for i,estimator in enumerate(tqdm(estimators)):
    m,spacex=scan1d(o,estimator,boundary=boundaries[0])
    matrices.append(m)

locationx=np.arange(n)[spacex>prior_sample[0]].min()
indexes_x=['%0.f'%f for f in spacex]
ticks=np.linspace(0,n-1,2).astype(np.int)
ax=plt.subplot(2,4,4)
ax.annotate(letters[l], xy=(-0.05, 1.15), xycoords='axes fraction', textcoords='offset points', fontsize=15,xytext=(0, -5), weight='bold', ha='right',  va='top')
l+=1
plt.locator_params(axis='y', nbins=4)
plt.title(model_names[-1],y=1.08)

for i,m in enumerate(matrices):
    plt.plot(m,label=objectives_names[i])
    plt.axvline(locationx,c='k')
    plt.xticks(ticks,[indexes_x[int(i)] for i in ticks])
#plt.legend(frameon=False,handlelength=1)   
plt.xlabel(r"$\rho$")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0),useMathText=True)

plt.tight_layout(pad=0.2)
plt.savefig('results/plots/fig3.pdf')