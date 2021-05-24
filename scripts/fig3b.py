import numpy as np
import pandas as pd
from mimsbi.divergences import MutualInformation
from mimsbi.generate import Generator
import matplotlib.pyplot as plt
from scripts.utils import *
from mimsbi.models.bd import Simulator, Prior
import matplotlib.cm as cm
import matplotlib as mpl
from scipy.stats import norm
plt.style.use('seaborn-white')

n_par=2
boundaries=[[-2,2.],[2.,50.]]
n_values=10
default_measurement_time=0.5
prior=Prior(scale_alpha=boundaries[0][1],scale_beta=boundaries[1][1])
simulator=Simulator(default_measurement_time=default_measurement_time,n_values=n_values)
seed=1234   
theta=np.array([0.,20.])

for i in range(5):
    
    plt.figure(figsize=(4,4),dpi=200)
    times,states=simulator.trajectory(theta)
    selection=simulator.return_selection(times)
    plt.plot(times,states,zorder=0,linewidth=1,color='C0')
    plt.scatter(np.array(times)[selection],states[selection],s=30,zorder=1,color='white',edgecolors='C0',linewidth=1.5)
    plt.xlabel('time')
    plt.ylabel('n')
    plt.savefig('results/plots/fig2b_'+str(i)+'.pdf')
    
n_simulations=int(1e5)
theta,x_= Generator(n_simulations,prior,simulator)

# normalise data
mean=x_.mean(axis=0)
std=x_.std(axis=0)
simulator.means=mean
simulator.stds=std
x=(x_-mean)/std

estimator=MutualInformation(values1=theta,values2=x,l2_reg=1e-5,objective='MINE',seed=seed,validation_split=0.1)
estimator.fit(epochs=5000,batch_size=1000,seed=seed,weights_name='results/weights/fig3b.h5')



for k in range(5):
    m_=[]
    for beta in [5,10,20,30,40]:
        prior_sample=np.array([0.,beta])
        observation=[simulator.simulate(prior_sample) for i in range(5)]
        m,spacex,spacey=scan2d(observation,estimator,boundaryx=[-2,2],boundaryy=[2,50])
        m_.append(m)


    plt.figure(figsize=(4,4),dpi=200)
    colors = iter(cm.rainbow(np.linspace(0, 1, 5)))

    for n,i in enumerate([5,10,20,30,40]):
        d=next(colors)
        plt.plot(spacey,m_[n].sum(axis=0),color=d)
        plt.axvline(i,color=d)
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'density')
    plt.savefig('results/plots/fig3b_'+str(k)+'.pdf')