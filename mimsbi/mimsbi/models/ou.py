from mimsbi.models.gillespie import Gillespie
import numpy as np    
import numpy.random as rng
from scipy.stats import uniform
from scipy.integrate import odeint

class Process:
    def __init__(self, theta,tau):
        self.sigma=theta[0]
        self.mu=theta[1]
        self.tau=tau
        self.sigma_bis = self.sigma * np.sqrt(2. / self.tau) # got rid of sqrt(2)
            
    def sim_time(self, duration,dt):
        """Simulates the process with the Euler-Maruyama method for a specified number of steps."""
        sqrtdt = np.sqrt(dt)
        times=np.arange(0,duration,dt)
        states = np.zeros(len(times))
        sq=self.sigma_bis * sqrtdt
        # simulate
        for i in range(len(times) - 1):
            states[i + 1] = states[i] + dt * (-(states[i] - self.mu) / self.tau) + sq * np.random.randn()
        return times, np.array(states)
        
class Simulator:

    def __init__(self, default_measurement_time=0.5, step_size=0.001,
                 means=None,stds=None,n_values=10,tau=1.):
        super(Simulator, self).__init__()
        
        if means is None: 
            self.means=np.array([0]*(n_values))
        else: self.means=means
        if stds is None: 
            self.stds=np.array([1]*(n_values))
        else: self.stds=stds
        self.default_measurement_time =default_measurement_time
        self.n_values=n_values
        self.step_size = float(step_size)
        self.tau=tau
        
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
        
    def calc_summary_stats(self, states,times,n_values=None):

        N = states.shape[0]
        x = states.copy()
        if n_values is None: n_values=self.n_values
        selection=self.return_selection(times,n_values)
        return (x[selection] -self.means)/self.stds
    
    def simulate(self, theta):
        self.mjp=Process(theta,self.tau)
        times,states = self.mjp.sim_time(self.default_measurement_time,self.step_size)
        sum_stats = self.calc_summary_stats(states,times)
        return sum_stats
    
    def trajectory(self, theta):
        self.mjp=Process(theta,self.tau)
        times,states = self.mjp.sim_time(self.default_measurement_time,self.step_size)
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

    def __init__(self,sigma=2.,mu=10.): 
        self.min=-1
        self.sigma=sigma 
        self.mu=mu 
        self.max=1
        self.u1 = uniform(0, self.max*self.sigma) 
        self.u2 = uniform(self.min*self.mu, (self.max-self.min)*self.mu )

    def sample(self, n=1):
        u1_samples = rng.uniform(0,self.max*self.sigma, n)
        u2_samples = rng.uniform(self.min*self.mu,self.max*self.mu, n)
        return np.array(list(zip(u1_samples,u2_samples)))
    
    def log_prob(self,theta): 
        return np.log(self.u1.pdf(theta[:,0]))+np.log(self.u2.pdf(theta[:,1]))

class Posterior:
    def __init__(self,simulator,true_shape=None): 
        self.dt = simulator.default_measurement_time/simulator.n_values
        self.tau=simulator.tau 
        self.means = simulator.means
        self.stds = simulator.stds
        self.true_shape = true_shape # take only xs not delta xs !
        
        self.u2 = 1-np.exp(-2*self.dt/self.tau)
        self.u1 = 1-np.exp(-self.dt/self.tau)
        self.u0 = 1-self.u1
        
    def rerenormalize(self,x):
        #if self.true_shape is None:
        #    self.true_shape = int(x.shape[1]+1)//2 
        #return x[:,:self.true_shape]*self.stds[:self.true_shape] + self.means[:self.true_shape]
        return x*self.stds + self.means

        
    def log_prob(self,data):
        sigma, mu, x_norm = data[:,0], data[:,1], data[:,2:]
        x = self.rerenormalize(x_norm)
        
        log_post = np.zeros(x.shape[0])
        for x0,xt in zip(x.T,x.T[1:]):
            mean = x0*self.u0 + mu*self.u1
            log_post -= (xt-mean)**2
        var = sigma**2*self.u2
        log_post = log_post/2/var - 0.5*len(x.T[1:])*np.log(2*np.pi*var)
        return log_post
    
    def compute_energy(self,data):
        return - self.log_prob(data)
        
    def log_ratio(self, values1, values2):
        return self.log_prob(np.concatenate([values1,values2],axis=1))