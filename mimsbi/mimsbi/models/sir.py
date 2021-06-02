"""

Created on 2 May 2021

@author: None

Gillespie simulation of SIR model. 
Based on code and structure from the software https://github.com/montefiore-ai/hypothesis

"""

from scipy.stats import uniform
from mimsbi.models.gillespie import Gillespie
import numpy as np    
import numpy.random as rng

class Process(Gillespie):
    """Implements the SIR model."""

    def _calc_propensities(self):
        S,I,R = self.state
        N=S+I+R
        return np.array([S*I/N, I])

    def _do_reaction(self, reaction):

        if reaction == 0:
            self.state[0] -= 1
            self.state[1] += 1
        elif reaction == 1:
            self.state[1] -= 1
            self.state[2] += 1
        else:
            raise ValueError('Unknown reaction.')    

class Simulator:

    def __init__(self, population_size=[97,3,0], default_measurement_time=50., step_size=1,
                 means=None,stds=None,n_values=10):
        
        super(Simulator, self).__init__()
        if means is None: 
            self.means=np.array([0]*(n_values*2))
        else: self.means=means
        if stds is None: 
            self.stds=np.array([1]*(n_values*2))
        else: self.stds=stds
        self.default_measurement_time =default_measurement_time
        self.n_values=n_values
        self.population_size = population_size
        self.step_size = float(step_size)

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

    def calc_summary_stats(self,states,times,norm=100,n_values=None):
        """
        Given a sequence of states produced by a simulation, calculates and returns a vector of summary statistics.
        Assumes that the sequence of states is uniformly sampled in time.
        """

        N = states.shape[0]
        x,y,z = states[:, 0].copy(), states[:, 1].copy(),states[:, 2].copy()
        if n_values is None: n_values=self.n_values
        selection=self.return_selection(times,n_values)
                
        out=np.concatenate([np.log(x[selection]+1),np.log(y[selection]+1)])
                                
        return (out-self.means)/self.stds

    def simulate(self, theta):

        self.mjp=Process(self.population_size, theta)
        times,states = self.mjp.sim_time(int(self.default_measurement_time))
        sum_stats = self.calc_summary_stats(states,times,np.sum(self.population_size))
        return sum_stats
    
    def trajectory(self, theta):

        self.mjp=Process(self.population_size, theta)
        times,states = self.mjp.sim_time(int(self.default_measurement_time))
        return times,states
    
    def forward(self, inputs):
        outputs = []

        n = len(inputs)
        for index in range(n):
            theta = inputs[index]
            x = self.simulate(theta)
            outputs.append(x)

        return np.array(outputs)
    
class Prior:

    def __init__(self):
        self.min=0.
        self.max=1.
        self.u1 = uniform(self.min, self.max-self.min)
        self.u2 = uniform(self.min, self.max-self.min)

    def sample(self, n=1):
        u1_samples = rng.uniform(self.min,self.max, n)
        u2_samples = rng.uniform(self.min,self.max, n)
        return np.array(list(zip(u1_samples,u2_samples)))
    
    def log_prob(self,x):
        return np.log(self.u1.pdf(x[:,0]))+np.log(self.u2.pdf(x[:,1]))
