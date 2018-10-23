# coding: utf-8

import abc
import numpy as np
np.seterr(all='raise', under='warn')

# nn model

class NNModel(abc.ABC):
    
    def step_fit(self, X, y):
        """Learn model with given input data set.
        
        Args:
            X (np.array): input data
            y (np.array): ground truth corresponding to X
        
        Returns:
            Total cost of samples of each round in running sequence (list)
        """
        y_predict = self.fit_forward(X)
        self.update_model(y_predict, y)
        return (self.calc_loss(y_predict, y)).sum()
    
    def fit(self, X, y, rounds, batch_size, random_shuffle=True):
        """Learn and update model with given input data set for specific rounds and batch size.
        
        Args:
            X (np.array): input data
            y (np.array): ground truth corresponding to X
        
        Returns:
            Average cost of samples of each round in running sequence (list)
        """
        avg_cost_by_round = []
        total_size = len(X)
        for _ in range(rounds):
            cost = 0
            learning_idx = np.arange(total_size)
            if random_shuffle:
                np.random.shuffle(learning_idx)

            for start_idx in range(0, total_size, batch_size):
                data_idx = learning_idx[start_idx : min(start_idx+batch_size,total_size)]
                cost += self.step_fit(X[data_idx], y[data_idx])
            avg_cost_by_round.append(cost/total_size)
        return avg_cost_by_round
        
    def transform(self, X):
        """Transform input data X into target data using current learned model.
        Support both batch and single sample.
        
        Args:
            X (np.array): input data
        
        Returns:
            Result calculated by current model
        """
        return self.forward(X)
    
    @abc.abstractmethod
    def fit_forward(self, X):
        """Forward pass, Run through each layers with given input X to get predict value.
        Support both batch and single sample.
        Intermediate result is stored for back propagation.
        
        Args:
            X (np.array): input data
        
        Returns:
            Result calculated by current model(all layers)
        """
        pass
    
    @abc.abstractmethod
    def forward(self, X):
        """Forward pass, run through each layers with given input X to get predict value.
        Support both batch and single sample.
        Intermediate result is not stored.
        
        Args:
            X (np.array): input data
        
        Returns:
            Result calculated by current model
        """
        pass
    
    @abc.abstractmethod
    def update_model(self, y_predict, y):
        """Update parameters of each layer using errs (y_predict - y) in current step.
        
        Args:
            y_predict (np.array): predict values
            y (np.array): target data
        """
        pass
    
    @abc.abstractmethod
    def calc_loss(self, y_predict, y):
        """Calculate cost value of current model of predict result.
        
        Args:
            y_predict (np.array): predict values
            y (np.array): target data
        
        Returns:
            Cost values of predict results by samples
        """
        pass


class VNNModel(NNModel):
    """Vanilla neural network."""
    
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
    
    def fit_forward(self, X):
        """Forward pass, Run through each layers with given input X to get predict value.
        Support both batch and single sample.
        Intermediate result is stored for back propagation.
        
        Args:
            X (np.array): input data
        
        Returns:
            Result calculated by current model(all layers)
        """
        cur_res = X
        for layer in self.model:
            cur_res = layer.fit_forward(cur_res)
        return cur_res
    
    def forward(self, X):
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
    
    def calc_loss(self, y_predict, y):
        """Calculate cost value of current model of predict result.
        
        Args:
            y_predict (np.array): predict values
            y (np.array): target data
        
        Returns:
            Cost values of predict results by samples
        """
        return self.cost_function.loss(y, y_predict)
