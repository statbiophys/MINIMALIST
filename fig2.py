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
np.random.seed(1)

c0='gold' 
c1='C9'
fig=plt.figure(figsize=(12,6),dpi=200)
for j,model in enumerate(models):

    objectives=['BCE']
    estimators=[]
    
    for obj in objectives: estimators.append(MutualInformation(load_dir='results/estimators/'+model+'_sim7_obj'+obj+'_rep0'))
        
    averages=np.load('results/data/averages_'+model+'.npy',allow_pickle=True).item()
    mean=averages['mean']
    std=averages['std']
    observations=pd.read_csv('results/data/data_prior_sample_'+model+'.csv.gz').values
    obs=(observations-mean)/std
    n_par,prior_sample,boundaries,default_measurement_time,prior,simulator= return_pars(model)
    
    ax=plt.subplot(2,4,1+j)
    ax.annotate(letters[l], xy=(-0.1, 1.15), xycoords='axes fraction', textcoords='offset points', fontsize=15,xytext=(0, -5), weight='bold', ha='right',  va='top')
    l+=1
    plt.title(model_names[j])

    plt.locator_params(axis='y', nbins=4)
    plt.locator_params(axis='x', nbins=4)

    if model=='sir':
        times,states=simulator.trajectory(prior_sample)
        selection=simulator.return_selection(times)
        plt.plot(times,states[:,0],color='C4',label='S')
        plt.plot(times,states[:,1],color=c1,label='I')
        plt.plot(times,states[:,2],color='darkgreen',label='R')
        plt.legend(frameon=False,handlelength=1)

        plt.gca().set_prop_cycle(None)
        plt.scatter(np.array(times)[selection],states[selection][:,0],zorder=10,color='white',edgecolors='C4',linewidth=1.5,facecolor='w')
        plt.scatter(np.array(times)[selection],states[selection][:,1],zorder=11,color='white',edgecolors=c1,linewidth=1.5,facecolor='w')
        #plt.scatter(np.array(times)[selection],states[selection][:,2],zorder=13,color='white',edgecolors=c1,linewidth=1.5,facecolor='w')
    else:
            times,states=simulator.trajectory(prior_sample)
            selection=simulator.return_selection(times)
            plt.plot(times,states,c=c1)
            plt.scatter(np.array(times)[selection],states[selection],zorder=10,color='white',edgecolors=c1,linewidth=1.5,facecolor='w')
            times,states=simulator.trajectory(prior_sample)
            selection=simulator.return_selection(times)
            plt.plot(times,states,c=c0)
            plt.scatter(np.array(times)[selection],states[selection],zorder=10,color='white',edgecolors=c0,linewidth=1.5,facecolor='w')
    plt.xlabel('time')
    plt.ylabel('n')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0),useMathText=True)

    if model=='ou':    plt.ylabel('x')
    
    matrices=[]
    o=obs[np.random.choice(np.arange(len(obs)),size=n_obs[model])]
    for i,estimator in enumerate(tqdm(estimators)):
        m,spacex,spacey=scan2d(o,estimator,boundaryx=boundaries[0],boundaryy=boundaries[1])
        matrices.append(m)
        
    

    locationy=np.arange(n)[spacey>prior_sample[1]].min()
    locationx=np.arange(n)[spacex>prior_sample[0]].min()
    indexes_x=['%0.f'%f for f in spacex]
    indexes_y=['%0.f'%f for f in spacey]
    ticks=np.linspace(0,n-1,2).astype(np.int)
    ax=plt.subplot(2,4,5+j)
    ax.annotate(letters[l], xy=(-0.1, 1.15), xycoords='axes fraction', textcoords='offset points', fontsize=15,xytext=(0, -5), weight='bold', ha='right',  va='top')
    l+=1
    plt.imshow(m)
    plt.yticks(ticks,[indexes_x[int(i)] for i in ticks])
    plt.xticks(ticks,[indexes_y[int(i)] for i in ticks])
    plt.ylabel(x_labels1[j])
    plt.xlabel(x_labels2[j])
    plt.scatter(locationy,locationx,c='r',s=20)
    if model=='ou':
                ax.annotate("true value",xy=(locationy,locationx),xytext=(locationy-50, locationx+30),xycoords='data', textcoords='data',
            arrowprops=dict(arrowstyle="->",connectionstyle="arc3"),
            )

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

ax = fig.add_subplot(2, 4, 4, projection='3d')
ax.annotate(letters[l], xy=(-0.1, 1.15), xycoords='axes fraction', textcoords='offset points', fontsize=15,xytext=(0, -5), weight='bold', ha='right',  va='top')
l+=1
plt.title(model_names[-1])
plt.locator_params(axis='y', nbins=4)

times,states=simulator.trajectory(prior_sample)
selection=simulator.return_selection(times)
plt.plot(xs=states[:,0],ys=states[:,1],zs=states[:,2],c=c1)
ax.scatter(xs=states[selection,0],ys=states[selection,1],zs=states[selection,2],edgecolors=c1,linewidth=1.5,zorder=10)
times,states=simulator.trajectory(prior_sample)
selection=simulator.return_selection(times)
plt.plot(xs=states[:,0],ys=states[:,1],zs=states[:,2],c=c0)
ax.scatter(xs=states[selection,0],ys=states[selection,1],zs=states[selection,2],edgecolors=c0,linewidth=1.5,zorder=10)
ax.set_axis_off()

matrices=[]
o=obs[np.random.choice(np.arange(len(obs)),size=5)]
for i,estimator in enumerate(tqdm(estimators)):
    m,spacex=scan1d(o,estimator,boundary=boundaries[0])
    matrices.append(m)

locationx=np.arange(n)[spacex>prior_sample[0]].min()
indexes_x=['%0.f'%f for f in spacex]
ticks=np.linspace(0,n-1,2).astype(np.int)
ax=plt.subplot(2,4,8)
ax.annotate(letters[l], xy=(-0.1, 1.15), xycoords='axes fraction', textcoords='offset points', fontsize=15,xytext=(0, -5), weight='bold', ha='right',  va='top')
l+=1
plt.locator_params(axis='y', nbins=4)


plt.plot(m,c='k')
plt.axvline(locationx,c='r')
plt.xticks(ticks,[indexes_x[int(i)] for i in ticks])
plt.xlabel(r"$\rho$")
plt.ylabel('Posterior Density')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0),useMathText=True)

plt.tight_layout(pad=0.1)
plt.savefig('results/plots/fig2.pdf')