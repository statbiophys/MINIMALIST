import numpy as np
from mimsbi.divergences import MutualInformation
from mimsbi.generate import Generator
from mimsbi.models.lorentzPar import Simulator,Prior
import matplotlib.pyplot as plt
import pandas as pd
from scripts.utils import *
import matplotlib.cm as cm
import matplotlib as mpl
from scipy.stats import norm
plt.style.use('seaborn-white')

n_simulations=int(1e5)
model='lorentz'
delta_lambda=0.5
n_values=10
noise=.5
default_measurement_time=delta_lambda*n_values*0.905
start=[-3.10330849, -1.55169067, 25.16791608]

#define prior and simulator
prior= Prior()
simulator=Simulator(default_measurement_time=default_measurement_time,n_values=n_values,noise=noise,start=start)

#generate data
n_simulations=int(1e5)
theta,x_= Generator(n_simulations,prior,simulator)
mean=x_.mean(axis=0)
std=x_.std(axis=0)
x=(x_-mean)/std
simulator.means=mean
simulator.stds=std
regularization,batch_size,learning_rate=return_hyperpars('MINE','lorentz')

#infer estimators
estimators=[]
for n in range(n_values):
    estimator=MutualInformation(values1=theta,values2=x[:,:n+1],objective='MINE',validation_split=0.1,l2_reg=regularization,lr=learning_rate)
    estimator.fit(epochs=5000,batch_size=batch_size,patience=40,weights_name='results/weights/fig3c.h5')
    estimators.append(estimator)
    
    
# plot inference curves
colors = cm.rainbow(np.linspace(0, 1, n_values))
plt.figure(figsize=(5,4),dpi=200)
for n,estimator in enumerate(estimators):
    plt.plot(-np.array(estimator.learning_history.history['val_likelihood'])*1.44,color=colors[n],linewidth=0.5)
plt.xlabel('epoch')
plt.ylabel('MI [bits]')
number_of_lines=n_values
cmap = mpl.cm.rainbow
bounds = np.arange(number_of_lines+1)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
sm = plt.cm.ScalarMappable(cmap='rainbow')
clb=plt.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow'),ticks=np.arange(number_of_lines)+0.5)
clb.ax.set_yticklabels([n+1 for n in range(n_values)])
clb.ax.set_xlabel(r'$n$')
plt.savefig('results/plots/fig3c_inference.pdf')



loc=35
x=np.linspace(30,40,100)

for i in range(5):
    results=[]
    for n in range(n_values):
        observation=[]
        for j in range(10):
            observation.append(simulator.simulate(np.array([[loc]])))
        m,space=scan1d(list(np.array(observation)[:,:n+1]),estimators[n],
                               boundary=[30.,40.])
        results.append(m)
    plt.figure(figsize=(4,4),dpi=200)
    for n in range(n_values):
        plt.plot(space,results[n],color=colors[n],linewidth=1)
    plt.axvline(35.,linewidth=0.5,color='k')
    plt.legend(frameon=False)
    clb=plt.colorbar(cm.ScalarMappable(norm=norm, cmap='rainbow'),ticks=np.arange(number_of_lines)+0.5)
    clb.ax.set_yticklabels([n+1 for n in range(n_values)])
    clb.ax.set_xlabel(r'$n$')
    plt.xlabel(r'$\rho$')
    plt.savefig('results/plots/fig3c_'+str(i)+'.pdf')
