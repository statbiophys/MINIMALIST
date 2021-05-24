import numpy as np
from mimsbi.divergences import MutualInformation
from mimsbi.generate import ConditionalGenerator,Generator
from mimsbi.models.lorentzPos import Simulator, Prior
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import matplotlib as mpl
from scipy.stats import norm
from scripts.utils import *
plt.style.use('seaborn-white')

default_measurement_time=14.57*1.2
n_values=1
noise=0.
prior= Prior(scale=.1)
simulator=Simulator(default_measurement_time=default_measurement_time,n_values=n_values,noise=noise)

times,states1=simulator.trajectory([0.])
times,states2=simulator.trajectory([1e-2])
plt.figure(figsize=(16,2),dpi=200)
selection=simulator.return_selection(times)
plt.plot(times,states1[:,0],linewidth=.5)
plt.plot(times,states1[:,1],linewidth=.5)
plt.plot(times,states1[:,2],linewidth=.5)
plt.scatter(np.array(times)[selection],states1[selection][:,0],c='C0',s=10)
plt.scatter(np.array(times)[selection],states1[selection][:,1],c='C1',s=10)
plt.scatter(np.array(times)[selection],states1[selection][:,2],c='C2',s=10)
plt.plot(times,states2[:,0],'--',linewidth=.5,c='C0')
plt.plot(times,states2[:,1],'--',linewidth=.5,c='C1')
plt.plot(times,states2[:,2],'--',linewidth=.5,c='C2')
plt.scatter(np.array(times)[selection],states2[selection][:,0],marker='s',c='C0',s=10)
plt.scatter(np.array(times)[selection],states2[selection][:,1],marker='s',c='C1',s=10)
plt.scatter(np.array(times)[selection],states2[selection][:,2],marker='s',c='C2',s=10)
colors = cm.viridis(np.linspace(0, 1, 4))
number_of_lines=4

for n,i in enumerate(14.57*np.array([0.6,0.8,1.,1.2])):
    plt.axvline(i,color=colors[n])
bounds = np.arange(number_of_lines+1)
cmap = mpl.cm.rainbow
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
sm = plt.cm.ScalarMappable(cmap='viridis')
clb=plt.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'),ticks=np.arange(number_of_lines)+0.5)
clb.ax.set_yticklabels([0.6,0.8,1.,1.2])
clb.ax.set_xlabel(r'$\lambda$')#plt.legend(frameon=False)
plt.savefig('results/plots/fig3d_trajectory.pdf')

# plot divergences 
diff=(states2-states1)
plt.figure(figsize=(4,4),dpi=200)
d=np.sqrt(np.sum(diff**2,axis=1))
x=np.linspace(0.,times.max(),100)
plt.semilogy(x,0.01*np.exp(0.9*x),label='$0.01*exp(0.9 t)$')
plt.semilogy(times,d,label='$\Delta(y1,y2)$')
plt.legend(frameon=True)
plt.savefig('results/plots/fig3d_divergence.pdf')

estimators=[]
simulators=[]
for i in 14.57*np.array([0.6,0.8,1.,1.2]):
    print(i)
    noise=0.01
    simulator=Simulator(default_measurement_time=i,n_values=1,noise=0.)
    n_simulations=int(1e5)
    theta,x_= Generator(n_simulations,prior,simulator)
    mean=x_.mean(axis=0)
    std=x_.std(axis=0)
    
    # add noise
    x=(x_*(1.+np.random.randn(len(x_)*3).reshape(len(x_),3)*noise)-mean)/std
    simulators.append(Simulator(default_measurement_time=i,n_values=1,noise=0.,means=mean,stds=std))
    estimator=MutualInformation(values1=theta,values2=x,l2_reg=1e-5,objective='sMINE',validation_split=0.1)
    estimator.fit(epochs=5000,batch_size=1000,weights_name='results/weights/fig3d_inferene.h5')
    estimators.append(estimator)


colors = cm.viridis(np.linspace(0, 1, 4))
plt.figure(figsize=(5,4),dpi=200)
 
for n,estimator in enumerate(estimators):
    plt.plot(-np.array(estimator.learning_history.history['val_likelihood'])*1.44,color=colors[n],linewidth=0.5)
plt.xlabel('epoch')
plt.ylabel('MI [bits]')
bounds = np.arange(number_of_lines+1)
cmap = mpl.cm.rainbow
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
sm = plt.cm.ScalarMappable(cmap='viridis')
clb=plt.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'),ticks=np.arange(number_of_lines)+0.5)
clb.ax.set_yticklabels([0.6,0.8,1.,1.2])
clb.ax.set_xlabel(r'$\lambda$')
plt.savefig('results/plots/fig3d_inference.pdf')


results=[]
prior_sample=np.array([0.,28.])
for k in range(5):
    
    for i in range(len(estimators)):
        observation=simulators[i].simulate(prior_sample)
        obs_=observation*simulators[i].stds+simulators[i].means    
        obs=(obs_*(1.+np.random.randn(3)*noise)-simulators[i].means)/simulators[i].stds
        m,spacex=scan1d([obs],estimators[i],boundary=[-0.1,0.1],n=200)
        results.append(m)

    x=np.linspace(-0.1,0.1,100)
    number_of_lines=4
    colors = cm.viridis(np.linspace(0, 1, number_of_lines))
    plt.figure(figsize=(5,4),dpi=200)
    for i in range(len(estimators)):
        plt.plot(spacex,results[i],color=colors[i],linewidth=1.5)
    #plt.axvline(prior_sample[0],linewidth=1,color='k')
    bounds = np.arange(number_of_lines+1)
    cmap = mpl.cm.rainbow
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    sm = plt.cm.ScalarMappable(cmap='viridis')
    clb=plt.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'),ticks=np.arange(number_of_lines)+0.5)
    clb.ax.set_yticklabels([0.6,0.8,1.,1.2])
    clb.ax.set_xlabel(r'$\lambda$')#plt.legend(frameon=False)
    plt.xlabel(r'$x_0$')
    plt.ylim([0.001,1.])
    plt.yscale('log')
    plt.savefig('results/plots/fig3d_'+str(k)+'.pdf')
