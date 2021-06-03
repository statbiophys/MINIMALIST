import numpy as np
from mimsbi.mcmc import NormalTransition,MCMC,Chain
import matplotlib.pyplot as plt
from mimsbi.divergences import MutualInformation,DklDivergence,DjsDivergence
from sklearn.metrics import roc_curve, auc


def return_pars(model):
    
    """
    
    parameters of the models for the benchmark
    
    """
    
    if model=='bd':
        from mimsbi.models.bd import Simulator, Prior
        n_par=2
        prior_sample=np.array([0.2, 10.])
        boundaries=[[-2,2.],[2.,20.]]
        n_values=10
        default_measurement_time=0.5
        prior=Prior(scale_alpha=boundaries[0][1],scale_beta=boundaries[1][1])
        simulator=Simulator(default_measurement_time=default_measurement_time,n_values=n_values)
    elif model=='sir':
        from mimsbi.models.sir import Simulator, Prior
        n_par=2
        n_values=10
        prior_sample=np.array([.6, 0.15])
        boundaries=[[0.,1.],[0.,1.]]
        default_measurement_time=50.
        prior=Prior()
        simulator=Simulator()
    elif model=='ou':
        from mimsbi.models.ou import Simulator, Prior
        n_par=2
        n_values=10
        default_measurement_time=0.5
        prior_sample=np.array([1., 5.],)
        boundaries=[[0.,2.],[-10,10]]
        prior=Prior()
        simulator=Simulator()
    elif model=='lorenz' or model=='lorentz':
        from mimsbi.models.lorentzPar import Simulator, Prior
        n_par=1
        n_values=5
        prior_sample=np.array([35])
        noise=.5
        delta_lambda=0.5
        boundaries=[[30.,40.]]
        default_measurement_time=delta_lambda*n_values*0.905
        start=[-3.10330849, -1.55169067, 25.16791608]
        
        prior=Prior()
        simulator=Simulator(default_measurement_time=default_measurement_time,
                            n_values=n_values,noise=noise,start=start)

    else:
        print('Wrong model.')
        exit()
    return n_par,prior_sample,boundaries,default_measurement_time,prior,simulator

def return_hyperpars(objective,model):
    d=np.load('results/data/hyperpars_'+model+'.npy',allow_pickle=True).item()       
    return d[objective]
    
def sigmoid(x):
    return 1./(1+np.exp(-x))

def plot_training(estimator,savename=None):
    plt.figure(figsize=(12,4),dpi=100)
    plt.subplot(131)
    plt.plot(estimator.likelihood_train,label='train',c='C0')
    plt.plot(estimator.likelihood_test,label='val',c='C1')
    plt.ylabel('MI(bits)', fontsize = 15)
    plt.xlabel('epoch')
    plt.legend(frameon=False)
    plt.subplot(132)
    plt.plot(estimator.BCE_train,label='train',c='C0')
    plt.plot(estimator.BCE_test,label='val',c='C1')
    plt.ylabel('BCE(bits)', fontsize = 15)
    plt.xlabel('epoch')
    plt.legend(frameon=False)
    plt.subplot(133)
    plt.hist(estimator.energies_numerator,100,histtype='step',density=True,label='numerator',color='C0')
    plt.hist(estimator.energies_denominator,100,histtype='step',density=True,label='denominator',color='C1')
    plt.ylabel('density', fontsize = 15)
    plt.xlabel('energy')
    plt.legend(frameon=False)
    plt.tight_layout()
    if savename is None:  
        plt.show()
    else:
        plt.savefig(savename)
        
def scan2d(observation,estimator,boundaryx=[0,1],boundaryy=[0,1],n=100):
    matrix_values=np.ones((n,n))
    spacex=np.linspace(boundaryx[0],boundaryx[1],n)
    spacey=np.linspace(boundaryy[0],boundaryy[1],n)

    to_evaluate=[]
    for ni,i in enumerate(spacex):
        for nj,j in enumerate(spacey):
            to_evaluate.append([np.concatenate([[i,j],o]) for o in observation])
    energies=[]
    for i in range(len(observation)):
        energies.append(estimator.compute_energy(np.array(to_evaluate)[:,i]))
    k=0
    for ni,i in enumerate(spacex):
        for nj,j in enumerate(spacey):
            for f in range(len(observation)):
                matrix_values[ni,nj]*=np.exp(-energies[f][k])
            k+=1
    return matrix_values/np.sum(matrix_values),spacex,spacey

def scan1d(observation,estimator,boundary=[0,1],n=100):
    matrix_values=np.ones((n))
    spacex=np.linspace(boundary[0],boundary[1],n)

    to_evaluate=[]
    for ni,i in enumerate(spacex):
        to_evaluate.append([np.concatenate([[i],o]) for o in observation])
    energies=[]
    for i in range(len(observation)):
        energies.append(estimator.compute_energy(np.array(to_evaluate)[:,i]))
    k=0
    for ni,i in enumerate(spacex):
        for f in range(len(observation)):
            matrix_values[ni]*=np.exp(-energies[f][k])
        k+=1
    return matrix_values/np.sum(matrix_values),spacex

def dkl(p1,p2):
    p1=p1/np.sum(p1)
    p2=p2/np.sum(p2)
    return np.sum(p1*(np.log2(p1+1e-50)-np.log2(p2+1e-50)))

def djs(p1,p2):
    m=(p1+p2)/2.
    return 0.5*(dkl(p1,m)+dkl(p2,m))

def sample_posterior(estimator,prior,observation,n_chain=100):
    ntheta=100
    theta = prior.sample(n=ntheta)
    #burning chain
    transition = NormalTransition(.5)
    mcmc = MCMC(prior, estimator, transition)
    burnin_chain = mcmc.sample(theta, observation, n_chain)
    print(burnin_chain.acc_ratio())
    theta0 = burnin_chain.best_theta(mcmc,observation)
    # select new thetas
    ntheta_new = 10000
    p = np.exp(burnin_chain.ratios)
    indices = np.random.choice(np.arange(ntheta),size=ntheta_new,replace=True,p=p/sum(p))
    theta2 = (burnin_chain.samples[-1])[indices]
    transition = NormalTransition(.1)
    mcmc = MCMC(prior, estimator, transition)
    chain = mcmc.sample(theta2, observation, n_chain)
    print(chain.acc_ratio())
    chain.trim()
    print(len(chain.samples_trimmed_flat))
    return chain.samples_trimmed_flat

def compare_samples(samples1,samples2,objective='BCE',nodes_number=10,validation_split=0.5,weights_name='weights.h5',n=10,patience=10,code=''):
        
    min_length=np.min([len(samples1),len(samples2)])
    samples1=samples1[:min_length]
    samples2=samples2[:min_length]

    #infer dkl
    dkl=DklDivergence(numerator=samples1,denominator=samples2,objective=objective,nodes_number=nodes_number,validation_split=validation_split)
    dkl.fit(epochs=1000,batch_size=1000,early_stopping=True,weights_name=weights_name,patience=patience) # to implement
    dklm,bcem=dkl.evaluate()
    
    plot_training(dkl,savename=code+'_dkl.png')
    
    djs=DjsDivergence(values1=samples1,values2=samples2,objective=objective,nodes_number=nodes_number, validation_split=validation_split)
    djs.fit(epochs=1000,batch_size=1000,early_stopping=True,weights_name=weights_name,patience=patience) # to implement
    djsm,djsbcm=djs.evaluate()

    plot_training(djs.dkl1,savename=code+'_djs1.png')
    plot_training(djs.dkl2,savename=code+'_djs2.png')
    
    fpr, tpr, _ = roc_curve(dkl.Y_test, sigmoid(dkl.compute_energy(dkl.X_test) + np.log(dkl.Z)))
    roc_auc = auc(fpr, tpr)
    
    return  dklm,bcem,djsm,djsbcm,roc_auc


