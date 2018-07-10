# coding: utf-8

import abc
import numpy as np
np.seterr(all='raise', under='warn')

# nn model


class NNLayer:
    """Single layer of neural network"""
    
    def __init__(self, node_count, has_intercepts, activation):
        """Init function.
        
        Attributes:
            layer_id (int/str): ID of current layer
            node_count (int): number of nodes of current layer
            has_intercepts (boolean): flag if the nodes of the layer has intercept terms
            activation (ActivationFunction): activation function
            gd_optimizer (GDOptimizer): gradient descent optimizer for GD update
            weights (np.array): input weights of all nodes
            intercepts (np.array): intercepts of all nodes
            weight_grads (np.array): weight gradients of all nodes of current step 
            intercept_grads (np.array): intercept gradients of all nodes of current step 
            step_input (np.array): input of current step
            reduced_sum (np.array): reduced sum of input before activation of current step 
            step_output (np.array): output of current step
        """
        self.layer_id = None
        self.node_count = node_count
        self.has_intercepts = has_intercepts
        self.activation = activation
        self.gd_optimizer = None
        self.weights = None
        self.intercepts = None
        self.weight_grads = None
        self.intercept_grads = None
        self.step_input = None
        self.reduced_sum = None
        self.step_output = None
    
    def init(self, layer_id, prev_node_count, gd_optimizer):
        """Initialize layer info and nodes.
        
        Args:
            layer_id (int/str): ID of current layer
            prev_node_count (int): number of input of current layer
            gd_optimizer (GDOptimizer): gradient descent optimizer for GD update
        """
        self.layer_id = layer_id
        self.gd_optimizer = gd_optimizer
        self.weights, self.intercepts = self.xavier_weight_init(prev_node_count, self.node_count, self.has_intercepts)
        self.gd_optimizer.register_layer(self.layer_id, self.weights.shape, self.intercepts.shape)
    
    def forward(self, step_input, learning=True):
        """Run through the layer with given input to get output.
        
        Args:
            step_input (np.array): input data
            learning (boolean): flag if current step is in learning period
                During learning period, intermediate result is stored for back propagation.
        
        Returns:
            Result calculated by current layer
            activation(input*weights+intercepts)
        """
        reduced_sum = np.dot(step_input, self.weights)+self.intercepts
        step_output = self.activation.apply(reduced_sum)
        if learning:
            self.step_input = step_input
            self.reduced_sum = reduced_sum
            self.step_output = step_output
        return step_output
    
    def back_propagation(self, prev_delta, activation_derivatived=False):
        """Back propagation to calculate gradient in last step for update.
        Store the gradients in weight_grads and intercept_grads.
        
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
        self.intercept_grads = cur_delta.mean(axis=0)
        return np.dot(cur_delta, self.weights.T)
    
    def update_weights(self):
        """Update weights of each input nodes using calculated gradients for last step
        Use pre-specified GD optimizer for accelerating convergance during GD update.
        """
        weight_deltas, intercept_deltas =             self.gd_optimizer.apply(self.layer_id, self.weight_grads, self.intercept_grads)
        self.weights -= weight_deltas
        if self.has_intercepts:
            self.intercepts -= intercept_deltas
    
    def xavier_weight_init(self, prev_node_count, cur_node_count, has_intercepts):
        """Initialize weight using Xavier init.
        
        Args:
            prev_node_count (int): number of input data (output of previous layer)
            cur_node_count (int): number of nodes (output of current layer)
            has_intercepts (boolean): flags if intercept terms are added
        
        Returns:
            Initialized weights and intercepts
        """
        weights = np.random.randn(prev_node_count, cur_node_count)/np.sqrt(prev_node_count)
        intercepts = np.random.randn(cur_node_count)/np.sqrt(prev_node_count)             if has_intercepts else np.zeros(cur_node_count)
        return weights, intercepts


class NNModel:
    def __init__(self, X_size, layers, gd_optimizer):
        """Init function.
        
        Attributes:
            X_size (int): number of input data
            model (list(NNLayer)): layers of current model
            gd_optimizer (GDOptimizer): gradient descent optimizer for GD update
        """
        self.X_size = X_size
        self.model = layers
        self.gd_optimizer = gd_optimizer
        self.init_model()
    
    def init_model(self):
        """Initialize each layer in the model.
        """
        prev_node_count = self.X_size
        for layer_no, layer in enumerate(self.model):
            layer.init(layer_no, prev_node_count, self.gd_optimizer)
            prev_node_count = layer.node_count
    
    def model_forward(self, X, learning=True):
        """Forward pass, Run through each layers with given input X to get predict value.
        Support both batch and single sample.
        Intermediate result is stored for back propagation if learning is set to True.
        
        Args:
            X (np.array): input data
            learning (boolean): flag if current step is in learning period
                During learning period, intermediate result is stored for back propagation.
        
        Returns:
            Result calculated by current model(all layers)
        """
        cur_res = X
        for layer in self.model:
            cur_res = layer.forward(cur_res, learning=learning)
        return cur_res
    
    def predict(self, X):
        """Forward pass, run through each layers with given input X to get predict value.
        Support both batch and single sample.
        Intermediate result is not stored.
        
        Args:
            X (np.array): input data
        
        Returns:
            Result calculated by current model
        """
        return self.model_forward(X, learning=False)
    
    def update_model(self, y_predict, y):
        """Update parameters of each layer using errs (y_predict - y) in current step.
        
        Args:
            y_predict (np.array): predict values
            y (np.array): target data
        """
        delta = np.atleast_2d(y_predict - y)
        for layer in reversed(self.model):
            delta = layer.back_propagation(delta, activation_derivatived=(layer==self.model[-1]))
        for layer in self.model:
            layer.update_weights()

