# coding: utf-8

import abc
import numpy as np
np.seterr(all='raise', under='warn')

from ..initializer import Initializer
from .components import activation as actv
from .components import core

# nn layer


class NNLayer(abc.ABC):
    """Base class for neural network layers."""
    
    def __init__(self, components):
        """Init function.
        
        Attributes:
            layer_id (int/str): ID of current layer
            components (list(LayerComponent)): Components which form the layer
        """
        self.layer_id = id(self)
        self.components = components
    
    def setup(self, **kwargs):
        """Init function during model creation."""
        for component in self.components:
            data = component.setup(**kwargs)
        return data
    
    def fit_forward(self, data):
        """Run through the layer with given input to get output for learning.
        
        Args:
            data (np.array): input data
        
        Returns:
            Result calculated by current layer
        """
        for component in self.components:
            data = component.fit_forward(data)
        return data
    
    def forward(self, data):
        """Run through the layer with given input to get output for actual calculation.
        
        Args:
            data (np.array): input data
        
        Returns:
            Result calculated by current layer
        """
        for component in self.components:
            data = component.forward(data)
        return data
    
    def back_propagation(self, cur_delta):
        """Back propagation to calculate gradient in last step for update.
        Store the gradients in weight_grads and bias_grads.
        
        Args:
            prev_delta (np.array): delta values(derivatives) from the next layer
                wrt output of current layer
            activation_derivatived (boolean): if the delta value contains
                derivative of activation function
        
        Returns:
            Derivatives of current layer wrt input
        """
        for component in reversed(self.components):
            cur_delta = component.back_propagation(cur_delta)
        return cur_delta
    
    def update(self):
        """Update weights of each input nodes using calculated gradients for last step.
        Use pre-specified GD updater for accelerating convergance during GD update.
        """
        for component in self.components:
            component.update()


class FCLayer(NNLayer):
    """Single layer of neural network"""
    
    def __init__(self, node_count,
                 has_bias=True,
                 activation=actv.ActivationNone(),
                 initializer=Initializer.xavier_normal):
        """Init function.
        
        Attributes:
            node_count (int): number of nodes of current layer
            has_bias (boolean): flag if the nodes of the layer has bias terms
            activation (ActivationFunction): activation function
            initializer (method): function to initialize weights
        """
        components = [
            core.Dense(node_count, has_bias=has_bias, initializer=initializer),
            activation,
        ]
        super(FCLayer, self).__init__(components)
        self.node_count = node_count
    
    def dropout(self):
        # TODO:
        pass
    
    def batch_normalization(self):
        # TODO:
        pass
        
