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
        self.weight_infos[layer_id] = np.zeros(weight_shape)
        self.bias_infos[layer_id] = np.zeros(bias_shape)
    
    @abc.abstractmethod
    def apply(self, layer_id, grads, bias_grads):
        """Calculate update value.
        
        Args:
            layer_id (int/str): key to specify the layer
            grads (np.array): gradients of weights
            bias_grads (np.array): gradients of bias terms
        
        Returns:
            Calculated gradients/update value
        """
        pass


class GDUpdaterNormal(GDUpdater):
    """No optimization, apply learning rate only."""
    
    def __init__(self, learning_rate):
        """Init function.
        
        Attributes:
            learning_rate (float): learning rate
        """
        GDUpdater.__init__(self)
        self.learning_rate = learning_rate
    
    def apply(self, layer_id, grads, bias_grads):
        """Calculate update value.
        
        Args:
            layer_id (int/str): key to specify the layer
            grads (np.array): gradients of weights
            bias_grads (np.array): gradients of bias terms
        
        Returns:
            learning_rate * gradients
        """
        return self.learning_rate * grads, self.learning_rate * bias_grads


# Need a smaller learning rate(0.01?)
class GDUpdaterMomentum(GDUpdater):
    """Momentum for SGD.
    
    In Momentum, weight_infos and bias_infos stores the last calculated weight deltas.
    """
    
    def __init__(self, learning_rate, gamma=0.9):
        """Init function.
        
        Attributes:
            learning_rate (float): learning rate
            gamma (float): rate of the update vector of the past time step to add on, default 0.9
        """
        GDUpdater.__init__(self)
        self.learning_rate = learning_rate
        self.gamma = gamma
    
    def apply(self, layer_id, grads, bias_grads):
        """Calculate update value.
        
        Args:
            layer_id (int/str): key to specify the layer
            grads (np.array): gradients of weights
            bias_grads (np.array): gradients of bias terms
        
        Returns:
            gamma * last_gradients + learning_rate * current_gradients
        """
        self.weight_infos[layer_id] = self.gamma*self.weight_infos[layer_id] + self.learning_rate*grads
        self.bias_infos[layer_id] = self.gamma*self.bias_infos[layer_id] + self.learning_rate*bias_grads
        return self.weight_infos[layer_id], self.bias_infos[layer_id]


class GDUpdaterAdagrad(GDUpdater):
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        GDUpdater.__init__(self)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
    
    def apply(self, layer_id, grads, bias_grads):
        # TODO
        pass


class GDUpdaterAdadelta(GDUpdater):
    def __init__(self, gamma=0.9, epsilon=1e-8):
        GDUpdater.__init__(self)
        self.gamma = gamma
        self.epsilon = epsilon
    
    def apply(self, layer_id, grads, bias_grads):
        # TODO
        pass


# TODO: Problems here, got a large error in the end and maybe not converged?
class GDUpdaterRMSprop(GDUpdater):
    """RMSprop.
    
    In RMSprop, weight_infos and bias_infos stores the last calculated E[g^2].
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
        
    def apply(self, layer_id, grads, bias_grads):
        """Calculate update value.
        
        Args:
            layer_id (int/str): key to specify the layer
            grads (np.array): gradients of weights
            bias_grads (np.array): gradients of bias terms
        
        Returns:
            E[g^2] = gamma * last_E[g^2] + (1-gamma) * current_gradients^2
            learning_rate * gradients / sqrt(E[g^2] + epsilon)
        """
        self.weight_infos[layer_id] = self.gamma*self.weight_infos[layer_id] \
            + (1-self.gamma)*np.power(grads,2)
        self.bias_infos[layer_id] = self.gamma*self.bias_infos[layer_id] \
            + (1-self.gamma)*np.power(bias_grads,2)
        weight_delta = self.learning_rate * grads / np.sqrt(self.weight_infos[layer_id] + self.epsilon)
        bias_delta = self.learning_rate * bias_grads / np.sqrt(self.bias_infos[layer_id] + self.epsilon)
        return weight_delta, bias_delta

