from __future__ import division
import numpy as np
import numpy.random as rng  

class SimTooLongException(Exception):

    def __init__(self, max_n_steps):
        self.max_n_steps = max_n_steps

    def __str__(self):
        return 'Simulation exceeded the maximum of {} steps.'.format(self.max_n_steps)

    
class Gillespie:
    """Implements a generic markov jump process with Gillespie.
    It is an abstract class, it needs to be inherited by a concrete implementation."""

    def __init__(self, init, params):

        self.state = np.asarray(init)
        self.params = np.asarray(params)
        self.time = 0.0

    def _calc_propensities(self):
        raise NotImplementedError('This is an abstract method and should be implemented in a subclass.')

    def _do_reaction(self, reaction):
        raise NotImplementedError('This is an abstract method and should be implemented in a subclass.')

    def sim_steps(self, num_steps):
        """Simulates the process with the Gillespie algorithm for a specified number of steps."""

        times = [self.time]
        states = [self.state.copy()]

        for _ in range(num_steps):

            rates = self.params * self._calc_propensities()
            total_rate = rates.sum()

            if total_rate == 0:
                self.time = float('inf')
                break

            self.time += rng.exponential(scale=1/total_rate)
            reaction= rng.choice(np.arange(len(rates)),p=rates / total_rate)  
            
            self._do_reaction(reaction)

            times.append(self.time)
            states.append(self.state.copy())

        return times, np.array(states)
    
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