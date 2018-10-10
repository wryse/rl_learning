# coding: utf-8

import abc
import numpy as np
np.seterr(all='raise', under='warn')

# gd updater


class GDUpdater(abc.ABC):
    """Base class for gradient descent updater. One updater to N layers."""
    
    def __init__(self):
        """Init function.
        
        Attributes:
            weight_infos (dict): buff to store any step result for weight
            bias_infos (dict): buff to store any step result for bias terms
        """
        self.weight_infos = {}
        self.bias_infos = {}
    
    def register_layer(self, layer_id, weight_shape, bias_shape):
        """Register layer which will use this updater.
        
        Args:
            layer_id (int/str): key to specify the layer
            weight_shape (int): shape of weight of the layer
            bias_shape (int): shape of bias terms of the layer
        """
        self.weight_infos[layer_id] = self.init_infos(weight_shape)
        self.bias_infos[layer_id] = self.init_infos(bias_shape)
    
    def apply(self, layer_id, grads, bias_grads):
        """Calculate update value.
        
        Args:
            layer_id (int/str): key to specify the layer
            grads (np.array): gradients of weights
            bias_grads (np.array): gradients of bias terms
        
        Returns:
            Calculated gradients/update value
        """
        grads_updated, self.weight_infos[layer_id] = self.calculate(grads, self.weight_infos[layer_id])
        bias_updated, self.bias_infos[layer_id] = self.calculate(bias_grads, self.bias_infos[layer_id])
        return grads_updated, bias_updated
    
    def init_infos(self, data_shape):
        """Initialize cache data (often zero).
        
        Args:
            data_shape (shape): shape of the data
        
        Returns:
            Initialized cache data
        """
        return np.zeros(data_shape)
    
    @abc.abstractmethod
    def calculate(self, grads, last_info):
        """Calculate update value.
        
        Args:
            grads (np.array): gradients of weights
            last_info (np.array): buffer value from previous steps
        
        Returns:
            Calculated gradients/update value
        """
        pass


class GDUpdaterNormal(GDUpdater):
    """No optimization, apply learning rate only."""
    
    def __init__(self, learning_rate=0.1):
        """Init function.
        
        Attributes:
            learning_rate (float): learning rate
        """
        GDUpdater.__init__(self)
        self.learning_rate = learning_rate
    
    def calculate(self, grads, last_info):
        """Calculate update value.
        
        Args:
            grads (np.array): gradients of weights
            last_info (np.array): buffer value from previous steps
        
        Returns:
            learning_rate * gradients
        """
        return self.learning_rate * grads, last_info


# Need a smaller learning rate(0.01?)
class GDUpdaterMomentum(GDUpdater):
    """Momentum for SGD.
    
    In Momentum, cache stores the last calculated weight deltas.
    """
    
    def __init__(self, learning_rate=0.01, gamma=0.9):
        """Init function.
        
        Attributes:
            learning_rate (float): learning rate
            gamma (float): rate of the update vector of the past time step to add on, default 0.9
        """
        GDUpdater.__init__(self)
        self.learning_rate = learning_rate
        self.gamma = gamma
    
    def calculate(self, grads, last_info):
        """Calculate update value.
        
        Args:
            grads (np.array): gradients of weights
            last_info (np.array): buffer value from previous steps
        
        Returns:
            gamma * last_result + learning_rate * current_gradients
        """
        cur_update = self.gamma*last_info + self.learning_rate*grads
        return cur_update, cur_update


class GDUpdaterAdagrad(GDUpdater):
    """Adagrad.
    
    In Adagrad, cache stores square of last gradients.
    """
    
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        """Init function.
        
        Attributes:
            learning_rate (float): learning rate
            epsilon (float): smoothing term that avoids division by zero, default 1e-8
        """
        GDUpdater.__init__(self)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
    
    def calculate(self, grads, last_info):
        """Calculate update value.
        
        Args:
            grads (np.array): gradients of weights
            last_info (np.array): buffer value from previous steps
        
        Returns:
            learning_rate * current_gradients / (sqrt(cache) + epsilon)
        """
        cur_info = last_info + np.power(grads, 2)
        cur_update = learning_rate * grads / (np.sqrt(cur_info) + self.epsilon)
        return cur_update, cur_info


class GDUpdaterAdadelta(GDUpdater):
    def __init__(self, gamma=0.9, epsilon=1e-8):
        GDUpdater.__init__(self)
        self.gamma = gamma
        self.epsilon = epsilon
    
    def apply(self, layer_id, grads, bias_grads):
        # TODO
        pass


class GDUpdaterRMSprop(GDUpdater):
    """RMSprop.
    
    In RMSprop, weight_infos and bias_infos stores the exponentially decaying average
    of past squared gradients E[g^2].
    """
    
    def __init__(self, learning_rate=0.001, gamma=0.9, epsilon=1e-8):
        """Init function.
        
        Attributes:
            learning_rate (float): learning rate, default 0.001
            gamma (float): rate of the update E[g^2] of the past time step to add on, default 0.9
            epsilon (float):  smoothing term that avoids division by zero, default 1e-8
        """
        GDUpdater.__init__(self)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
    def calculate(self, grads, last_info):
        """Calculate update value.
        
        Args:
            grads (np.array): gradients of weights
            last_info (np.array): buffer value from previous steps
        
        Returns:
            E[g^2] = gamma * last_E[g^2] + (1-gamma) * current_gradients^2
            learning_rate * gradients / sqrt(E[g^2] + epsilon)
        """
        cur_info = self.gamma*last_info + (1-self.gamma)*np.power(grads,2)
        cur_update = self.learning_rate * grads / np.sqrt(cur_info + self.epsilon)
        return cur_update, cur_info


class GDUpdaterAdam(GDUpdater):
    """Adam.
    
    In Adam, weight_infos and bias_infos stores the exponentially decaying average of past gradients m
    and squared gradients v
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """Init function.
        
        Attributes:
            learning_rate (float): learning rate, default 0.001
            beta1 (float): rate of the update past gradients to add on, default 0.9
            beta2 (float): rate of the update past squared gradients to add on, default 0.999
            epsilon (float):  smoothing term that avoids division by zero, default 1e-8
        """
        GDUpdater.__init__(self)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
    def init_infos(self, data_shape):
        """Initialize cache data.
        
        In Adam, two caches are needed.
        For this, rewrite cache init function.
        
        Args:
            data_shape (shape): shape of the data
        
        Returns:
            Initialized cache data
        """
        return (np.zeros(data_shape), np.zeros(data_shape))
    
    def calculate(self, grads, last_info):
        """Calculate update value.
        
        Args:
            grads (np.array): gradients of weights
            last_info (np.array): buffer value from previous steps
        
        Returns:
            m = beta1 * last_m + (1-beta1) * current_gradients
            v = beta2 * last_v + (1-beta2) * current_gradients^2
            learning_rate * m / sqrt(v + epsilon)
        """
        last_m, last_v = last_info
        m = self.beta1 * last_m + (1-self.beta1) * grads
        v = self.beta2 * last_v + (1-self.beta2) * np.power(grads,2)
        cur_update = self.learning_rate * m / np.sqrt(v + self.epsilon)
        return cur_update, (m, v)

