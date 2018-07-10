# coding: utf-8

import abc
import numpy as np
np.seterr(all='raise', under='warn')

# gd optimizer


class GDOptimizer(abc.ABC):
    """Base class for gradient descent optimizer. One optimizer to N layers."""
    
    def __init__(self):
        """Init function.
        
        Attributes:
            weight_infos (dict): buff to store any step result for weight
            intercept_infos (dict): buff to store any step result for intercept
        """
        self.weight_infos = {}
        self.intercept_infos = {}
    
    def register_layer(self, layer_id, weight_shape, intercept_shape):
        """Register layer which will use this optimizer.
        
        Args:
            layer_id (int/str): key to specify the layer
            weight_shape (int): shape of weight of the layer
            intercept_shape (int): shape of intercept of the layer
        """
        self.weight_infos[layer_id] = np.zeros(weight_shape)
        self.intercept_infos[layer_id] = np.zeros(intercept_shape)
    
    @abc.abstractmethod
    def apply(self, layer_id, grads, intercept_grads):
        """Apply optimizer to current gradients for update.
        
        Args:
            layer_id (int/str): key to specify the layer
            grads (np.array): gradients of weight
            intercept_grads (np.array): gradients of intercept
        
        Returns:
            optimized gradients/update value
        """
        pass


class GDOptimizerNone(GDOptimizer):
    """Do no optimization, apply learning rate only."""
    
    def __init__(self, learning_rate):
        """Init function.
        
        Attributes:
            learning_rate (float): learning rate
        """
        GDOptimizer.__init__(self)
        self.learning_rate = learning_rate
    
    def apply(self, layer_id, grads, intercept_grads):
        """Apply optimizer to current gradients for update.
        
        Args:
            layer_id (int/str): key to specify the layer
            grads (np.array): gradients of weight
            intercept_grads (np.array): gradients of intercept
        
        Returns:
            learning_rate * gradients
        """
        return self.learning_rate * grads, self.learning_rate * intercept_grads


# Need a smaller learning rate(0.01?)
class GDOptimizerMomentum(GDOptimizer):
    """Momentum for SGD.
    
    In Momentum, weight_infos and intercept_infos stores the last calculated weight deltas.
    """
    
    def __init__(self, learning_rate, gamma=0.9):
        """Init function.
        
        Attributes:
            learning_rate (float): learning rate
            gamma (float): rate of the update vector of the past time step to add on, default 0.9
        """
        GDOptimizer.__init__(self)
        self.learning_rate = learning_rate
        self.gamma = gamma
    
    def apply(self, layer_id, grads, intercept_grads):
        """Apply optimizer to current gradients for update.
        
        Args:
            layer_id (int/str): key to specify the layer
            grads (np.array): gradients of weight
            intercept_grads (np.array): gradients of intercept
        
        Returns:
            gamma * last_gradients + learning_rate * current_gradients
        """
        self.weight_infos[layer_id] = self.gamma*self.weight_infos[layer_id] + self.learning_rate*grads
        self.intercept_infos[layer_id] = self.gamma*self.intercept_infos[layer_id] + self.learning_rate*intercept_grads
        return self.weight_infos[layer_id], self.intercept_infos[layer_id]


class GDOptimizerAdagrad(GDOptimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        GDOptimizer.__init__(self)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
    
    def apply(self, layer_id, grads, intercept_grads):
        # TODO
        pass


class GDOptimizerAdadelta(GDOptimizer):
    def __init__(self, gamma=0.9, epsilon=1e-8):
        GDOptimizer.__init__(self)
        self.gamma = gamma
        self.epsilon = epsilon
    
    def apply(self, layer_id, grads, intercept_grads):
        # TODO
        pass


# TODO: Problems here, got a large error in the end and maybe not converged?
class GDOptimizerRMSprop(GDOptimizer):
    """RMSprop.
    
    In RMSprop, weight_infos and intercept_infos stores the last calculated E[g^2].
    """
    
    def __init__(self, learning_rate=0.001, gamma=0.9, epsilon=1e-8):
        """Init function.
        
        Attributes:
            learning_rate (float): learning rate, default 0.001
            gamma (float): rate of the update E[g^2] of the past time step to add on, default 0.9
            epsilon (float):  smoothing term that avoids division by zero, default 1e-8
        """
        GDOptimizer.__init__(self)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
    def apply(self, layer_id, grads, intercept_grads):
        """Apply optimizer to current gradients for update.
        
        Args:
            layer_id (int/str): key to specify the layer
            grads (np.array): gradients of weight
            intercept_grads (np.array): gradients of intercept
        
        Returns:
            E[g^2] = gamma * last_E[g^2] + (1-gamma) * current_gradients^2
            learning_rate * gradients / sqrt(E[g^2] + epsilon)
        """
        self.weight_infos[layer_id] = self.gamma*self.weight_infos[layer_id]             + (1-self.gamma)*np.power(grads,2)
        self.intercept_infos[layer_id] = self.gamma*self.intercept_infos[layer_id]             + (1-self.gamma)*np.power(intercept_grads,2)
        weight_delta = self.learning_rate * grads / np.sqrt(self.weight_infos[layer_id] + self.epsilon)
        intercept_delta = self.learning_rate * intercept_grads / np.sqrt(self.intercept_infos[layer_id] + self.epsilon)
        return weight_delta, intercept_delta

