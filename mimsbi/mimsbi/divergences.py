from mimsbi.density_estimator import DensityEstimator
import numpy as np

class DklDivergence(DensityEstimator):
    def __init__(self, numerator = [], denominator = [], load_dir = None, 
                 l2_reg = 0., l1_reg=0., nodes_number=50, objective='MINE',
                 seed=None,validation_split=0.1,optimizer='RMSprop'):
        super().__init__(numerator = numerator, denominator = denominator, load_dir = load_dir, 
                         l2_reg = l2_reg, l1_reg=l1_reg, nodes_number=nodes_number, 
                         objective=objective,seed=seed,validation_split=validation_split,optimizer=optimizer)

    # check if you want on train or test
    def get_value(self,n=10):
        return np.mean(self.likelihood_test[-n:]),np.std(self.likelihood_test[-n:])
    
    def get_bce(self,n=10):
        return np.mean(self.BCE_test[-n:]),np.std(self.BCE_test[-n:])
    
    def evaluate(self):
        _,MI,BCE=np.array(self.model.evaluate(self.X_test,self.Y_test,batch_size=10000))*1.44
        return -MI, BCE
    
class DjsDivergence(object):
    
    def __init__(self, values1 = [], values2 = [], load_dir = None, 
                 l2_reg = 0., l1_reg=0., nodes_number=50, objective='MINE',
                 seed=None,validation_split=0.1,optimizer='RMSprop'):
        
        n_num=len(values1)
        shuffle=np.random.choice(np.arange(2*n_num),2*n_num)
        # only same size of num and denominator are now implemented.
        self.m=np.concatenate([values1,values2])[shuffle][:n_num]
        
        self.dkl1=DklDivergence(numerator = values1, denominator = self.m, load_dir = load_dir, 
                                l2_reg = l2_reg, l1_reg=l1_reg, nodes_number=nodes_number, objective=objective,
                                seed=seed,validation_split=validation_split,optimizer=optimizer)
        
        self.dkl2=DklDivergence(numerator = values2, denominator = self.m, load_dir = load_dir, 
                                l2_reg = l2_reg, l1_reg=l1_reg, nodes_number=nodes_number, objective=objective,
                                seed=seed,validation_split=validation_split,optimizer=optimizer)
        
    def fit(self, epochs = 10, batch_size=5000, seed = None, verbose=0, callbacks=None,weights_name='weights.h5',early_stopping=False,patience=5):
        
        self.dkl1.fit(epochs = epochs, batch_size=batch_size, seed = seed, verbose=verbose, callbacks=callbacks,weights_name=weights_name,early_stopping=early_stopping,patience=patience)
        self.dkl2.fit(epochs = epochs, batch_size=batch_size, seed = seed, verbose=verbose, callbacks=callbacks,weights_name=weights_name,early_stopping=early_stopping,patience=patience)
        
    def get_value(self,n=10):
            
        d1m,d1s=self.dkl1.get_value(n)
        d2m,d2s=self.dkl2.get_value(n)
        return 0.5*(d1m+d2m),np.sqrt(d1s**2+d2s**2)
    
    def evaluate(self):
        MI1,BCE1=self.dkl1.evaluate()
        MI2,BCE2=self.dkl2.evaluate()
        return 0.5*(MI1+MI2), 0.5*(BCE1+BCE2)
    
    
class MutualInformation(DensityEstimator):
    
    def __init__(self, values1 = [], values2 = [], load_dir = None, 
                 l2_reg = 0., l1_reg=0., nodes_number=50, objective='MINE',
                 seed=None,validation_split=0.1,optimizer='RMSprop',gamma=0.001,lr=0.001):
        
        self.values1 = values1
        self.values2 = values2
        self.L1_converge_history = []
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.likelihood_train=[]
        self.likelihood_test=[]
        self.optimizer_=optimizer
        self.objective=objective
        self.nodes_number = nodes_number 
        self.gamma=gamma
        self.lr=lr
        self.Z=1.
        self.validation_split=validation_split
        if seed is not None: np.random.seed(seed = seed)
        
        if not load_dir is None:
            self.load_model(load_dir = load_dir)
        else:
            self.prepare_data() 
            self.update_model_structure(initialize=True)
    
    def add_data(self,values1=[],values2=[],validation_split=None):
        self.values1 = values1
        self.values2 = values2
        if not validation_split is None: 
            self.validation_split=validation_split
        self.prepare_data() 

    def prepare_data(self):
        n_simulations=len(self.values1)
        shuffling=np.random.choice(np.arange(n_simulations),n_simulations)

        self.numerator=np.concatenate([self.values1,self.values2],axis=1)
        self.denominator=np.concatenate([self.values1,self.values2[shuffling]],axis=1)

        x=np.concatenate([self.numerator,self.denominator],axis=0)
        y=np.zeros(2*n_simulations)
        y[n_simulations:]=1
        
        shuffle=np.random.choice(np.arange(2*n_simulations),2*n_simulations)
        self.X=x[shuffle][int(self.validation_split*n_simulations):]
        self.Y=y[shuffle][int(self.validation_split*n_simulations):]
        self.X_test=x[shuffle][:int(self.validation_split*n_simulations)]
        self.Y_test=y[shuffle][:int(self.validation_split*n_simulations)]
        
    def log_ratio(self, values1, values2):
        return -self.compute_energy(np.concatenate([values1,values2],axis=1))-np.log(self.Z)
    
    def get_value(self,n=10):
        return np.mean(self.likelihood_train[-n:]),np.std(self.likelihood_train[-n:])
    
    def get_value_max(self,n=10):
        return np.max(self.likelihood_train)
    
    def get_bce(self,n=10):
        return np.mean(self.BCE_train[-n:]),np.std(self.BCE_train[-n:])
    
    def evaluate(self,values1,values2):
        n_simulations=len(values1)
        shuffling=np.random.choice(np.arange(n_simulations),n_simulations)

        numerator=np.concatenate([values1,values2],axis=1)
        denominator=np.concatenate([values1,values2[shuffling]],axis=1)

        x__=np.concatenate([numerator,denominator],axis=0)
        y=np.zeros(2*n_simulations)
        y[n_simulations:]=1
        shuffling=np.random.choice(np.arange(2*n_simulations),2*n_simulations)
        _,MI,BCE=np.array(self.model.evaluate(x__[shuffling],y[shuffling],batch_size=10000))*1.44
        return -MI,BCE