"""

Created on 2 May 2021

@author: None

Gillespie simulation of birth death model. 
Based on code and structure from the software https://github.com/montefiore-ai/hypothesis

"""


from mimsbi.models.gillespie import Gillespie
import numpy as np    
import numpy.random as rng
from scipy.stats import uniform

class Process(Gillespie):
    """Implements the birth-death model."""

    def _calc_propensities(self):

        x = self.state[0]
        return np.array([x,x])

    def _do_reaction(self, reaction):

        if reaction == 0:
            self.state[0] += 1
        elif reaction == 1:
            self.state[0] -= 1
        else:
            raise ValueError('Unknown reaction.')   
            
    def sim_time(self, duration):
        """Simulates the process with the Gillespie algorithm for a specified number of steps."""

        times = [self.time]
        states = [self.state.copy()]

        while self.time < duration:

            rates = self.params * self._calc_propensities()
            total_rate = rates.sum()

            if total_rate == 0:
                self.time = float('inf')
                return times, np.array(states)

            self.time += rng.exponential(scale=1/total_rate)
            reaction= rng.choice(np.arange(len(rates)),p=rates / total_rate)  
            
            self._do_reaction(reaction)

            times.append(self.time)
            states.append(self.state.copy())

        return times, np.array(states)
        
class Simulator:

    def __init__(self, population_size=100, default_measurement_time=0.5, step_size=1,
                 means=None,stds=None,n_values=10):
        super(Simulator, self).__init__()
        if means is None: 
            self.means=np.array([0]*(n_values+n_values-1))
        else: self.means=means
        if stds is None: 
            self.stds=np.array([1]*(n_values+n_values-1))
        else: self.stds=stds
        self.default_measurement_time =default_measurement_time
        self.population_size = int(population_size)
        self.n_values=n_values
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
    
    def calc_summary_stats(self, states,times,norm=100,n_values=None):

        N = states.shape[0]
        x = states[:, 0].copy()
        if n_values is None: n_values=self.n_values
        selection=self.return_selection(times,n_values)
        
        logs=np.log(x[selection]+1)
        diff= np.diff(logs)
        out=np.concatenate([logs,diff])
        return (out- self.means)/self.stds
    
    def simulate(self, theta):
        theta_=np.array([theta[0]+theta[1],theta[1]-theta[0]])/2.
        self.mjp=Process([self.population_size], theta_)
        times,states = self.mjp.sim_time(self.default_measurement_time)
        sum_stats = self.calc_summary_stats(states,times,self.population_size)
        return sum_stats
    
    def trajectory(self, theta):
        theta_=np.array([theta[0]+theta[1],theta[1]-theta[0]])/2.
        self.mjp=Process([self.population_size], theta_)
        times, states = self.mjp.sim_time(self.default_measurement_time)
        return times,states
    
    def forward(self, inputs):
        outputs = []

        n = len(inputs)
        for index in range(n):
            theta = inputs[index]
            x = self.simulate(theta)
            outputs.append(x.view(1, -1))
        return outputs
    
class Prior:

    def __init__(self,scale_alpha=1.,scale_beta=10.):
        self.min=-1
        self.max=1
        self.scale_alpha=scale_alpha
        self.scale_beta=scale_beta
        self.u1 = uniform(self.min*self.scale_alpha, (self.max-self.min)*self.scale_alpha) # why - self.min
        self.u2 = uniform(self.max*self.scale_alpha, self.max*self.scale_beta-self.max*self.scale_alpha)

    def sample(self, n=1):
        u1_samples = rng.uniform(self.min*self.scale_alpha,self.max*self.scale_alpha, n)
        u2_samples = rng.uniform(self.max*self.scale_alpha,self.max*self.scale_beta, n)
        return np.array(list(zip(u1_samples,u2_samples)))
    
    def log_prob(self,x):
        return np.log(self.u1.pdf(x[:,0]))+np.log(self.u2.pdf(x[:,1]))
