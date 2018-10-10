# coding: utf-8

import abc
import numpy as np
np.seterr(all='raise', under='warn')

# nn model


class NNModel:
    def __init__(self, X_size, gd_updater, cost_function):
        """Init function.
        
        Attributes:
            X_size (int): number of input data
            gd_updater (GDUpdater): gradient descent updater for GD update
            cost_function (CostFunction): cost function
            model (list(NNLayer)): layers of current model
        """
        self.X_size = X_size
        self.gd_updater = gd_updater
        self.cost_function = cost_function
        self.model = []
    
    def add_layer(self, layer):
        """Add one layer to the model.
        
        Args:
            layer (NNLayer): Layer to be added
        """
        input_size = self.X_size
        if self.model:
            input_size = self.model[-1].node_count
        layer.setup(input_size=input_size, gd_updater=self.gd_updater)
        self.model.append(layer)
    
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
            cur_res = layer.fit_forward(cur_res)
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
        cur_res = X
        for layer in self.model:
            cur_res = layer.forward(cur_res)
        return cur_res
    
    def update_model(self, y_predict, y):
        """Update parameters of each layer using errs (y_predict - y) in current step.
        
        Args:
            y_predict (np.array): predict values
            y (np.array): target data
        """
        delta = np.atleast_2d(self.cost_function.derivative(y, y_predict))
        for layer in reversed(self.model):
            delta = layer.back_propagation(delta)
        for layer in self.model:
            layer.update()
    
    def loss(self, y_predict, y):
        """Calculate cost value of current model of predict result.
        
        Args:
            y_predict (np.array): predict values
            y (np.array): target data
        
        Returns:
            Cost values of predict results by samples
        """
        return self.cost_function.loss(y, y_predict)

