# coding: utf-8

import abc
import numpy as np
np.seterr(all='raise', under='warn')

from . import activation_function

# nn layer


class NNLayer(abc.ABC):
    """Base class for neural network layers."""
    
    def __init__(self):
        """Init function. Nothing needs to do in the base class."""
        pass
    
    def init(self):
        """Init function during model creation. Nothing needs to do in the base class."""
        pass
    
    @abc.abstractmethod
    def forward(self, step_input, learning=True):
        """Run through the layer with given input to get output.
        
        Args:
            step_input (np.array): input data
            learning (boolean): flag if current step is in learning period
                During learning period, intermediate result is stored for back propagation.
        
        Returns:
            Result calculated by current layer
            activation(input*weights+bias)
        """
        pass
    
    @abc.abstractmethod
    def back_propagation(self, prev_delta, activation_derivatived=False):
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
        pass
    
    @abc.abstractmethod
    def update(self):
        """Update weights of each input nodes using calculated gradients for last step.
        Use pre-specified GD updater for accelerating convergance during GD update.
        """
        pass
    
    def xavier_weight_init(self, prev_node_count, cur_node_count, has_bias):
        """Initialize weight using Xavier init.
        
        Args:
            prev_node_count (int): number of input data (output of previous layer)
            cur_node_count (int): number of nodes (output of current layer)
            has_bias (boolean): flags if bias terms are added
        
        Returns:
            Initialized weights and bias terms
        """
        weights = np.random.randn(prev_node_count, cur_node_count)/np.sqrt(prev_node_count)
        bias = np.random.randn(cur_node_count)/np.sqrt(prev_node_count) \
            if has_bias else np.zeros(cur_node_count)
        return weights, bias


class FCLayer(NNLayer):
    """Single layer of neural network"""
    
    def __init__(self, node_count, has_bias=True, activation=activation_function.ActivationNone()):
        """Init function.
        
        Attributes:
            layer_id (int/str): ID of current layer
            node_count (int): number of nodes of current layer
            has_bias (boolean): flag if the nodes of the layer has bias terms
            activation (ActivationFunction): activation function
            gd_updater (GDUpdater): gradient descent updater
            weights (np.array): input weights of all nodes
            bias (np.array): bias of all nodes
            weight_grads (np.array): weight gradients of all nodes of current step 
            bias_grads (np.array): bias gradients of all nodes of current step 
            step_input (np.array): input of current step
            reduced_sum (np.array): reduced sum of input before activation of current step 
            step_output (np.array): output of current step
        """
        self.layer_id = None
        self.node_count = node_count
        self.has_bias = has_bias
        self.activation = activation
        self.gd_updater = None
        self.weights = None
        self.bias = None
        self.weight_grads = None
        self.bias_grads = None
        self.step_input = None
        self.reduced_sum = None
        self.step_output = None
    
    def init(self, prev_node_count, gd_updater):
        """Init function during model creation. Initialize layer info and nodes.
        
        Args:
            prev_node_count (int): number of input of current layer
            gd_updater (GDUpdater): gradient descent updater
        """
        self.layer_id = id(self)
        self.gd_updater = gd_updater
        self.weights, self.bias = self.xavier_weight_init(prev_node_count, self.node_count, self.has_bias)
        self.gd_updater.register_layer(self.layer_id, self.weights.shape, self.bias.shape)
    
    def forward(self, step_input, learning=True):
        """Run through the layer with given input to get output.
        
        Args:
            step_input (np.array): input data
            learning (boolean): flag if current step is in learning period
                During learning period, intermediate result is stored for back propagation.
        
        Returns:
            Result calculated by current layer
            activation(input*weights+bias)
        """
        reduced_sum = np.dot(step_input, self.weights) + self.bias
        step_output = self.activation.apply(reduced_sum)
        if learning:
            self.step_input = step_input
            self.reduced_sum = reduced_sum
            self.step_output = step_output
        return step_output
    
    def back_propagation(self, prev_delta, activation_derivatived=False):
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
        cur_delta = None
        if activation_derivatived:
            cur_delta = prev_delta
        elif self.activation.derivative_use_activated:
            cur_delta = prev_delta * self.activation.derivative(self.step_output)
        else:
            cur_delta = prev_delta * self.activation.derivative(self.reduced_sum)
        self.weight_grads = np.dot(np.atleast_2d(self.step_input).T, cur_delta)/cur_delta.shape[0]
        self.bias_grads = cur_delta.mean(axis=0)
        return np.dot(cur_delta, self.weights.T)
    
    def update(self):
        """Update weights of each input nodes using calculated gradients for last step.
        Use pre-specified GD updater for accelerating convergance during GD update.
        """
        weight_deltas, bias_deltas = \
            self.gd_updater.apply(self.layer_id, self.weight_grads, self.bias_grads)
        self.weights -= weight_deltas
        if self.has_bias:
            self.bias -= bias_deltas
