# coding: utf-8

import abc
import numpy as np
np.seterr(all='raise', under='warn')

from ...initializer import Initializer
from .base import LayerComponent


# core layer components


class Dense(LayerComponent):
    
    def __init__(self, node_count, has_bias=True, initializer=Initializer.xavier_normal):
        """Init function.
        
        Attributes:
            node_count (int): number of nodes of current layer
            has_bias (boolean): flag if the nodes of the layer has bias terms
            initializer (method): function to initialize weights
            weights (np.array): input weights of all nodes
            bias (np.array): bias of all nodes
            weight_grads (np.array): weight gradients of all nodes of current step 
            bias_grads (np.array): bias gradients of all nodes of current step 
            gd_updater (GDUpdater): gradient descent updater
        """
        super(Dense, self).__init__()
        self.node_count = node_count
        self.has_bias = has_bias
        self.initializer = initializer
        self.weights = None
        self.bias = None
        self.weight_grads = None
        self.bias_grads = None
        self.gd_updater = None
    
    def setup(self, **kwargs):
        """Setup runtime parameters.
        
        Args:
            gd_updater (GDUpdater): gradient descent updater
            input_size (int): number of input of current layer
        """
        input_size = kwargs['input_size']
        self.gd_updater = kwargs['gd_updater']
        self.weights, self.bias = self.initializer((input_size, self.node_count), self.has_bias)
        self.gd_updater.register_layer(self.c_id, self.weights.shape, self.bias.shape)
    
    def forward(self, v):
        """Forward data.
        
        Args:
            v (np.array): Data
        
        Returns:
            Forward results
        """
        return np.dot(v, self.weights) + self.bias
    
    def back_propagation(self, prev_delta):
        """Back propagation to calculate gradient in last step for update.
        Store the gradients in weight_grads and bias_grads.
        
        Args:
            prev_delta (np.array): delta values(derivatives) from the next layer
                wrt output of current layer
        
        Returns:
            Derivatives of current layer wrt input
        """
        self.weight_grads = np.dot(np.atleast_2d(self.input).T, prev_delta)/prev_delta.shape[0]
        self.bias_grads = prev_delta.mean(axis=0)
        return np.dot(prev_delta, self.weights.T)
    
    def update(self):
        """Update weights of each input nodes using calculated gradients for last step.
        Use pre-specified GD updater for accelerating convergance during GD update.
        """
        weight_deltas, bias_deltas = \
            self.gd_updater.apply(self.c_id, self.weight_grads, self.bias_grads)
        self.weights -= weight_deltas
        if self.has_bias: self.bias -= bias_deltas

