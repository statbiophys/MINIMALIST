import numpy as np
import numpy.random as rng
from scipy.stats import uniform, norm as normal
from itertools import product

import matplotlib.pyplot as plt

def rotate(dim,seed=0):
    """ Random rotation in dimension dim """

    rng.seed(seed)
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        N = rng.normal(size=(dim-n+1,))
        D[n-1] = np.sign(N[0])
        N[0] -= D[n-1]*np.sqrt((N*N).sum())
        # Householder transformation
        Hn = (np.eye(dim-n+1) - 2.*np.outer(N, N)/(N*N).sum())
        mat = np.eye(dim)
        mat[n-1:, n-1:] = Hn
        H = np.dot(H, mat)
    D[-1] = (-1)**(1-(dim % 2))*D.prod()
    H = (D*H.T).T
    rng.seed()
    return H

class Prior:
    """ Prior over eigenvalues """
    
    def __init__(self,dim,lower=0.,upper=2.): 
        self.dim = dim
        self.upper = upper
        self.lower = lower
        self.u = uniform(self.lower, self.upper-self.lower) 
        
    def sample(self, n=1):
        return rng.uniform(self.lower,self.upper,size=self.dim*n).reshape(n,self.dim)
    
    def log_prob(self,theta): 
        return np.log(self.u.pdf(theta)).sum(axis=1)

class EigenProcess:
    """ Simulates the process with the Euler-Maruyama method for a specified duration """

        def __init__(self, theta, dim, rot_drag, rot_noise, tau, sigma=1., x0=None):
        self.dim = dim
        self.tau = tau
        self.rot_drag = rot_drag
        self.gamma = self.rot_drag.dot(np.diag(theta).dot(self.rot_drag.T))
        self.sigma = np.diag([sigma]*self.dim)
        if x0 is None: self.x0 = np.ones(self.dim)
        else: self.x0 = x0

    def sim_time(self, duration, dt):
        sqrtdt = np.sqrt(dt)
        times = np.arange(0,duration,dt)
        states = np.zeros((len(times),self.dim)) 
        states[0] = self.x0
        noise = np.sqrt(2./self.tau) * sqrtdt * self.sigma
        for i in range(len(times) - 1):
            states[i + 1] = states[i] -  self.gamma.dot(states[i]) * dt/self.tau + noise.dot(np.random.randn(self.dim))
        return times, np.array(states)
    
class Simulator:

    def __init__(self, dim, rot, default_measurement_time=0.2, step_size=0.001, n_values=10, 
                 tau=0.5, sigma=0.1, x0=None,
                 means=None, stds=None):
        super(Simulator, self).__init__()
        self.dim = dim
        self.default_measurement_time = default_measurement_time
        self.step_size = float(step_size)
        self.n_values = n_values
        self.tau = tau
        self.sigma = sigma
        self.x0 = x0
        self.rot = rot
        if means is None: 
            self.means=np.zeros(n_values*dim)
        else: self.means=means
        if stds is None: 
            self.stds=np.ones(n_values*dim)
        else: self.stds=stds
        
    def __del__(self):
        self.terminate()

    def terminate(self):
        pass
    
    def return_selection(self,times,n_values=None):
        if n_values is None: n_values=self.n_values
        selection=[]
        for i in range(n_values):
            try:
                selection.append(np.where((np.array(times)*n_values/self.default_measurement_time).astype(np.int)==i)[0].max())
            except:
                selection.append(selection[-1])
        return selection
        
    def calc_summary_stats(self,states,times,n_values=None):
        x = states.copy()
        if n_values is None: n_values=self.n_values
        selection=self.return_selection(times,n_values)
        return (x[selection].flatten()-self.means)/self.stds
    
    def simulate(self, theta):
        self.process = EigenProcess(theta, self.dim, self.rot, self.tau, self.sigma, self.x0)
        times,states = self.process.sim_time(self.default_measurement_time,self.step_size)
        sum_stats = self.calc_summary_stats(states,times)
        return sum_stats
    
    def trajectory(self, theta):
        self.process = EigenProcess(theta, self.dim, self.rot, self.tau, self.sigma, self.x0)
        times,states = self.process.sim_time(self.default_measurement_time,self.step_size)
        return times,states
    
    def test_rotation(self, theta, rot=None):
        if rot is None: rot = self.rot
        gammas = []
        for t in theta:
            process = EigenProcess(t, self.dim, rot, self.tau, self.sigma, self.x0)
            gammas.append(process.gamma.flatten())
        self.gs = np.array(gammas)
        plt.figure(figsize=(5,5))
        for g in self.gs.T:
            plt.hist(g,100,density=True,histtype='step')
        plt.show()

def vector_dot_vector(vectors1,vectors2):
    """ First index preserved """
    return np.einsum(vectors1, [...,1], vectors2, [...,1])

def matrix_dot_vector(matrices,vectors):
    """ First index preserved """
    return np.einsum(matrices, [...,1,2], vectors, [...,1])

def matrix_dot_matrix(matrices1,matrices2):
    """ First index preserved """
    return np.einsum(matrices1, [...,1,2], matrices2, [...,1,2])

class Posterior:
    def __init__(self,simulator): 
        # Inherit simulator features
        self.dim = simulator.dim
        self.rot = simulator.rot
        self.n_values = simulator.n_values
        self.dt = simulator.default_measurement_time/simulator.n_values
        self.tau = simulator.tau
        self.sigma2 = simulator.sigma**2
        
        self.means = simulator.means
        self.stds = simulator.stds
            
    def rerenormalize(self,x):
        # Go back to original x
        return (x * self.stds + self.means).reshape(x.shape[0],self.n_values,self.dim)
        
    def log_prob(self,data):
        nb_instances = len(data)
        gamma, x_norm = data[:,:self.dim], data[:,self.dim:]
        x = self.rerenormalize(x_norm)
        # define mean evolution operator
        evo = np.zeros((nb_instances,dim,dim))
        diagonal = np.arange(dim)
        evo[:,diagonal,diagonal] = np.exp(-gamma*self.dt/self.tau)
        evo_matrix = np.matmul(self.rot,np.matmul(evo,self.rot.T)) # back to x basis       
        # define covariance operator
        cov_inv = np.zeros((nb_instances,dim,dim))
        cov_inv[:,diagonal,diagonal] = gamma / self.sigma2 / (1 - np.exp(-2*gamma*self.dt/self.tau))
        cov_inv_matrix = np.matmul(self.rot,np.matmul(cov_inv,self.rot.T)) # back to x basis 
        cov_det = np.linalg.det(cov_inv_matrix)
        #cov_inv = np.linalg.inv(cov)   
        # compute log-probabilities
        log_post = np.zeros(nb_instances)
        x_transposed = np.moveaxis(x,1,0)
        for x0,xt in zip(x_transposed,x_transposed[1:]):
            mean = matrix_dot_vector(evo_matrix,x0)            
            log_post -= vector_dot_vector(xt-mean,matrix_dot_vector(cov_inv_matrix,xt-mean))/2
        log_post = log_post / len(x_transposed[1:])
        log_post -= 0.5*np.log(2*np.pi*cov_det)
        return log_post

    def log_ratio(self, values1, values2):
        return self.log_prob(np.concatenate([values1,values2],axis=1))

""" Important if sigma non-uniform """
...
#cov = np.zeros((nb_instances,self.dim,self.dim))
#for i,j in product(diagonal,repeat=2):
#    cov[:,i,j] = (1 - np.exp(-(gamma[:,i]+gamma[:,j])*self.dt/self.tau)) / (gamma[:,i]+gamma[:,j])
#cov *= self.sigma2
#cov_det = np.linalg.det(cov)
#cov_inv = np.matmul(self.rot,np.matmul(np.linalg.inv(cov),self.rot.T)) # back to x basis        