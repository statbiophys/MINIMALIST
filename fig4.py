import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use('seaborn-white')
from sys import argv
import os
import matplotlib as mlp
mlp.rcParams.update({'font.size': 14})

hue_order = ['MINE', 'fMINE', 'BCE']
model_names=['Ornstein–Uhlenbeck','Birth-Death','SIR','Lorenz attractor']
def plotit(obj):
    ax = sns.violinplot(
    x="sims", y=obj, hue="objective",
    data=dfs_, palette="muted",cut=0, inner=None,edgecolor='white',linewidth=0.5,hue_order=hue_order)
    handles = ax.legend_.legendHandles
    labels = ['MINE', 'FDIV', 'BCE']

    g=sns.swarmplot(data=dfs_, x="sims", y=obj, hue="objective",s=2,dodge=True,edgecolor='black',linewidth=0.5,hue_order=hue_order)
    plt.legend(handles, labels,handlelength=1)
    plt.ylabel(obj)
    plt.xlabel('')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0),useMathText=True)
    left, right =plt.ylim()
    plt.ylim([0,right+0.1*right])

letters=list('ABCDEFGHIJKLMNO')
plt.figure(figsize=(12,9),dpi=200)
models=['ou','bd','sir','lorentz']
l=0
for j,model in enumerate(models):

    results=os.listdir('results/benchmarks/')
    results=[e for e in results if model in e]

    dfs_=[]
    for i in [4,5,6]:
        dfs=[]
        r=[e for e in results if 'sim'+str(i) in e]
        for e in r:
            dfs.append(pd.read_csv('results/benchmarks/'+e))
        dfs=pd.concat(dfs)
        dfs['sims']=r'$10^{'+str(i)+'}$'
        dfs_.append(dfs)
    dfs_=pd.concat(dfs_)
    
    ax=plt.subplot(3,4,1+j)
    ax.annotate(letters[l], xy=(-0.1, 1.15), xycoords='axes fraction', textcoords='offset points', fontsize=15,xytext=(0, -5), weight='bold', ha='right',  va='top')
    l+=1
    plt.title(model_names[j])
    plotit('MI')
    plt.ylabel('mutual information [bits]')
    if model!='ou':    
        plt.ylabel('')
        plt.legend([],[], frameon=False)
    
    ax.spines['bottom'].set_position('zero')
    ax=plt.subplot(3,4,5+j)
    ax.annotate(letters[l], xy=(-0.1, 1.15), xycoords='axes fraction', textcoords='offset points', fontsize=15,xytext=(0, -5), weight='bold', ha='right',  va='top')
    l+=1
    plotit('djs_scan')
    plt.ylabel(r'$D_{JS}(Posteriors)$ [bits]')
    if model!='ou':    plt.ylabel('')
    plt.legend([],[], frameon=False)    
    ax.spines['bottom'].set_position('zero')
    ax=plt.subplot(3,4,9+j)
    ax.annotate(letters[l], xy=(-0.1, 1.15), xycoords='axes fraction', textcoords='offset points', fontsize=15,xytext=(0, -5), weight='bold', ha='right',  va='top')
    l+=1
    plotit('djs_mcmc_like')
    plt.ylabel(r'$D_{JS}(Likelihoods)$ [bits]')
    if model!='ou':    plt.ylabel('')
    plt.xlabel('N')
    plt.legend([],[], frameon=False)
    ax.spines['bottom'].set_position('zero')
plt.tight_layout(pad=1.1)
plt.savefig('results/plots/fig4.pdf')