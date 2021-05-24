#!/usr/bin/env python
# coding: utf-8

import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from mimsbi.divergences import MutualInformation,DklDivergence,DjsDivergence
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sys import argv
from tqdm import tqdm
from utils import *
plt.style.use('seaborn-white')


model=argv[1]
n_simulations=int(argv[2])
objective=str(argv[3])
n_obs={'sir':2,'ou':5,'bd':5,'lorentz':5}


reps=10
seeds=np.arange(reps).astype(np.int)
n_par,prior_sample,boundaries,default_measurement_time,prior,simulator= return_pars(model)

#take the average of infinite data
objectives=['MINE','fMINE','BCE']
a_samples_infinite=[]

averages=np.load('results/data/averages_'+model+'.npy',allow_pickle=True).item()
mean=averages['mean']
std=averages['std']
observations=pd.read_csv('results/data/data_prior_sample_'+model+'.csv.gz').values
obs=(observations-mean)/std
accepted_samples_infinite=obs[np.random.choice(len(obs),size=int(5e4))]

results=[]
for rep in tqdm(range(reps)):
    code='{}_sim{}_obj{}_rep{}'.format(model,int(np.log10(n_simulations)),objective,rep).replace('.','_')

    # infinite data
    ms_infinite=[]
    s_infinite=[]
    np.random.seed(seeds[rep])
    observation=obs[np.random.choice(np.arange(len(obs)),size=n_obs[model])]
    for obj in objectives:
        infinite_code='{}_sim{}_obj{}_rep{}'.format(model,int(np.log10(1e7)),obj,0).replace('.','_')

        estimator_infinite=MutualInformation(load_dir='results/estimators/'+infinite_code)
        samples_infinite=sample_posterior(estimator_infinite,prior,observation,n_chain=200)

        if not model=='lorentz':
            m_infinite,spacex,spacey=scan2d(observation,estimator_infinite,boundaryx=boundaries[0],boundaryy=boundaries[1])
        else:
            m_infinite,spacex=scan1d(observation,estimator_infinite,boundary=boundaries[0])
        if model=='lorentz':
            plt.figure(figsize=(4,4),dpi=200)
            a,b,c=plt.hist(samples_infinite,spacex,density=True,histtype='step',color='C1',label='inf samples')
            plt.plot(spacex,m_infinite/np.diff(b)[0],color='C1',label='inf scan')
            plt.legend(frameon=False)
            plt.xlabel('parameter 1')
            plt.savefig('results/estimators/'+infinite_code+'/posterior_scan.png')
        else:
            plt.figure(figsize=(8,4),dpi=200)
            plt.subplot(121)
            a,b,c=plt.hist(samples_infinite[:,0],spacex,density=True,histtype='step',color='C1',label='inf samples')
            plt.plot(spacex,m_infinite.sum(axis=1)/np.diff(b)[0],color='C1',label='inf scan')
            plt.legend(frameon=False)
            plt.xlabel('parameter 1')
            plt.subplot(122)
            a,b,c=plt.hist(samples_infinite[:,1],spacey,density=True,histtype='step',color='C1',label='inf samples')
            plt.plot(spacey,m_infinite.sum(axis=0)/np.diff(b)[0],color='C1',label='inf scan')
            plt.legend(frameon=False)
            plt.xlabel('parameter 2')
            plt.savefig('results/estimators/'+infinite_code+'/posterior_scan.png')
            

        s_infinite.append(samples_infinite)
        ms_infinite.append(m_infinite)

    #results of the average
    m_infinite=np.array(ms_infinite).mean(axis=0)
    samples_infinite=np.concatenate(s_infinite)
    np.random.shuffle(samples_infinite)
    
     # load estimator
    estimator=MutualInformation(load_dir='results/estimators/'+code)
    accepted_samples=pd.read_csv('results/estimators/'+code+'/accepted_samples.csv.gz').values
    samples=sample_posterior(estimator,prior,observation,n_chain=200)
    
    if not model=='lorentz':
        m,spacex,spacey=scan2d(observation,estimator,boundaryx=boundaries[0],boundaryy=boundaries[1])
    else:
        m,spacex=scan1d(observation,estimator,boundary=boundaries[0])
        
    if model=='lorentz':
        plt.figure(figsize=(4,4),dpi=200)
        plt.hist(samples,spacex,density=True,histtype='step',color='C0',label='est samples')
        a,b,c=plt.hist(samples_infinite,spacex,density=True,histtype='step',color='C1',label='inf samples')
        plt.plot(spacex,m/np.diff(b)[0],color='C0',label='est scan')
        plt.plot(spacex,m_infinite/np.diff(b)[0],color='C1',label='inf scan')
        plt.legend(frameon=False)
        plt.xlabel('parameter 1')
        plt.savefig('results/estimators/'+code+'/posterior_scan.png')
    else:
        plt.figure(figsize=(8,4),dpi=200)
        plt.subplot(121)
        plt.hist(samples[:,0],spacex,density=True,histtype='step',color='C0',label='est samples')
        a,b,c=plt.hist(samples_infinite[:,0],spacex,density=True,histtype='step',color='C1',label='inf samples')
        plt.plot(spacex,m.sum(axis=1)/np.diff(b)[0],color='C0',label='est scan')
        plt.plot(spacex,m_infinite.sum(axis=1)/np.diff(b)[0],color='C1',label='inf scan')
        plt.legend(frameon=False)
        plt.xlabel('parameter 1')
        plt.subplot(122)
        plt.hist(samples[:,1],spacey,density=True,histtype='step',color='C0',label='est samples')
        a,b,c=plt.hist(samples_infinite[:,1],spacey,density=True,histtype='step',color='C1',label='inf samples')
        plt.plot(spacey,m.sum(axis=0)/np.diff(b)[0],color='C0',label='est scan')
        plt.plot(spacey,m_infinite.sum(axis=0)/np.diff(b)[0],color='C1',label='inf scan')
        plt.legend(frameon=False)
        plt.xlabel('parameter 2')
        plt.savefig('results/estimators/'+code+'/posterior_scan.png')
    
    # first comparison of posteriors
    dkl1=dkl(m,m_infinite)
    dkl2=dkl(m_infinite,m)
    djs1=djs(m,m_infinite)
    
    # posterior comparison
    dklm,bcem,djsm,djsbcm,roc_auc=compare_samples(samples_infinite,samples,weights_name='results/weights/'+code+'e.h5',
                                                  code='results/estimators/'+code+'/posterior')
    # likelihood comparison
    dklm_l,dkls_l,djsm_l,djss_l,roc_auc_l=compare_samples(accepted_samples_infinite,accepted_samples,weights_name='results/weights/'+code+'e.h5',
                                                          code='results/estimators/'+code+'/likelihood')
    
    results.append([rep,objective,estimator.bce_value,estimator.mi_value,
                    dkl1,dkl2,djs1,dklm,bcem,djsm,djsbcm,
                    roc_auc,dklm_l,dkls_l,djsm_l,djss_l,roc_auc_l])
    
df=pd.DataFrame(results,columns=['rep','objective','BCE','MI','dkl_scan_inf_den','dkl_scan_inf_num',
                                 'djs_scan','dkl_mcmc_post','bce_dkl_mcmc_post','djs_mcmc_post','djs_bce_mcmc_post',
                                 'auc_mcmc_post','dkl_mcmc_like','bce_dkl_mcmc_like','djs_mcmc_like','djs_bce_mcmc_like',
                                 'auc_mcmc_like'])

code='{}_sim{}_obj{}'.format(model,int(np.log10(n_simulations)),objective).replace('.','_')
df.to_csv('results/benchmarks/'+code+'.csv.gz',compression='gzip',index=False)