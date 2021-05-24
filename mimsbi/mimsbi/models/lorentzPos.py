from mimsbi.models.gillespie import Gillespie
import numpy as np    
import numpy.random as rng
from scipy.stats import uniform
from scipy.integrate import solve_ivp

def lorentz(t,state,sigma,beta,rho):
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z 

class Process:
    def __init__(self, theta,noise,start):
        self.initial_pos=theta[0]
        self.rho=28.
        self.sigma=10.  
        self.beta=8./3.
        self.noise=noise
        self.start=start
        
    def sim_time(self, duration,dt):
        initial_position=self.start+np.array([self.initial_pos, 0., 0.])
        sol = solve_ivp(lorentz, [0, duration], initial_position, args=(self.sigma, self.beta, self.rho),dense_output=True)
        times=np.arange(0,duration,dt)
        states=sol.sol(times).T
        return times, np.array(states)
        
class Simulator:

    def __init__(self, default_measurement_time=50., step_size=0.005,
                 means=None,stds=None,n_values=5,noise=0.,start=np.array([-5,-4,24])):
        super(Simulator, self).__init__()
        
        if means is None: 
            self.means=np.array([0]*(n_values*3))
        else: self.means=means
        if stds is None: 
            self.stds=np.array([1]*(n_values*3))
        else: self.stds=stds
        self.default_measurement_time =default_measurement_time
        self.n_values=n_values
        self.step_size = float(step_size)
        self.noise=noise
        self.start=start
                            
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

        x,y,z = states[:, 0].copy(), states[:, 1].copy(),states[:, 2].copy()
        if n_values is None: n_values=self.n_values
        selection=self.return_selection(times,n_values)
        out=np.concatenate([x[selection],y[selection],z[selection]])  
        out=out+np.random.randn(len(out))*self.noise
        return (out-self.means)/self.stds
    
    def simulate(self, theta):
        self.mjp=Process(theta,self.noise,self.start)
        times,states = self.mjp.sim_time(self.default_measurement_time,self.step_size)
        sum_stats = self.calc_summary_stats(states,times)
        return sum_stats
    
    def trajectory(self, theta):
        self.mjp=Process(theta,self.noise,self.start)
        times,states = self.mjp.sim_time(self.default_measurement_time,self.step_size)
        return times,states
    
class Prior:

    def __init__(self,scale=.1):
        self.min=-1
        self.max=1
        self.scale=scale
        self.u1 = uniform(self.min*self.scale, (self.max-self.min)*self.scale) # why - self.min

    def sample(self, n=1):
        return rng.uniform(self.min*self.scale,self.max*self.scale, n).reshape(n,1)
    
    def log_prob(self,x):
        return np.log(self.u1.pdf(x))
