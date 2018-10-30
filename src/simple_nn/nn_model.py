# coding: utf-8

import abc
import numpy as np
np.seterr(all='raise', under='warn')

# nn model

class NNModelBase(abc.ABC):
    
    def fit(self, X, y, epoch=1, batch_size=None, random_shuffle=True):
        """Learn and update model with given input data set for specific rounds and batch size.
        
        Args:
            X (np.array): input data
            y (np.array): ground truth corresponding to X
        
        Returns:
            Average cost of samples of each round in running sequence (list)
        """
        avg_cost_by_epoch = []
        total_size = len(X)
        batch_size = total_size if batch_size is None else batch_size
        for _ in range(epoch):
            cost = 0
            learning_idx = np.arange(total_size)
            if random_shuffle:
                np.random.shuffle(learning_idx)

            for start_idx in range(0, total_size, batch_size):
                data_idx = learning_idx[start_idx : min(start_idx+batch_size,total_size)]
                cost += self.step_fit(X[data_idx], y[data_idx])
            avg_cost_by_epoch.append(cost/total_size)
        return avg_cost_by_epoch
        
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
    def step_fit(self, X, y):
        """Learn model with given input data set.
        
        Args:
            X (np.array): input data
            y (np.array): ground truth corresponding to X
        
        Returns:
            Total cost of samples of each round in running sequence (list)
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

class NNModel(NNModelBase):
    """Vanilla neural network."""
    
    def __init__(self, X_shape, gd_updater, cost_function):
        """Init function.
        
        Attributes:
            X_shape (int/shape): number or shape of input data
            gd_updater (GDUpdater): gradient descent updater for GD update
            cost_function (CostFunction): cost function
            model (list(NNLayer)): layers of current model
        """
        self.X_shape = X_shape
        self.gd_updater = gd_updater
        self.cost_function = cost_function
        self.model = []
    
    def step_fit(self, X, y):
        """Learn model with given input data set.
        
        Args:
            X (np.array): input data
            y (np.array): ground truth corresponding to X
        
        Returns:
            Total cost of samples of each round in running sequence (list)
        """
        y_predict = self.fit_forward(X)
        self.back_propagation(y_predict, y)
        self.update_model()
        return (self.calc_loss(y_predict, y)).sum()
    
    def add_layer(self, layer):
        """Add one layer to the model.
        
        Args:
            layer (NNLayer): Layer to be added
        """
        input_shape = self.X_shape
        if self.model:
            input_shape = self.model[-1].node_count
        layer.setup(input_size=input_shape, gd_updater=self.gd_updater)
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
    
    def back_propagation(self, y_predict, y, delta=None):
        """Back propagation to calculate gradient in last step.
        Either estimated result-ground truth pair or delta from layer underneath should be set.
        Store the gradients in weight_grads and bias_grads for update.
        
        Args:
            y_predict (np.array): predict values
            y (np.array): target data
            delta (np.array): delta from layer underneath
        """
        if delta is None:
            delta = np.atleast_2d(self.cost_function.derivative(y, y_predict))
        for layer in reversed(self.model):
            delta = layer.back_propagation(delta)
        return delta
    
    def update_model(self):
        """Update parameters of each layer using pre-calculated values."""
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


class GANModel(NNModel):
    """Generative adversarial network."""
    
    def random_noise(self, n, shape=None, spec_param=None):
        """Default function to totally generate noise randomly.
        
        Args:
            n (int): sample number to generate
            shape (int/shape): number or shape of single sample data
            spec_param (any): specific parameter for specific dataset, not used in the default function
        
        Returns:
            (Generated sample array, specific parameter array)
        """
        noise = np.random.randn(n, *shape)
        param = None
        return noise, param
    
    def __init__(self, gen_model, dis_model, func_gen_noise=None):
        """Init function.
        
        Attributes:
            gen_model (NNModel): generative model
            dis_model (NNModel): discriminal model
            Z_shape (int/shape): number or shape of input data for generative model
            X_shape (int/shape): number or shape of input data for discriminal model
            gen_noise (function): function of generating random noise with specific form of parameters
        """
        self.gen_model = gen_model
        self.dis_model = dis_model
        self.Z_shape = self.gen_model.X_shape
        self.X_shape = self.dis_model.X_shape
        self.gen_noise = func_gen_noise if func_gen_noise is not None else self.random_noise
    
    def step_fit(self, X, y):
        """Train the two models with given input sample data set.
        
        Args:
            X (np.array): input sample data
            y (np.array): ground truth corresponding to X
        
        Returns:
            Result of discriminal model as total cost in running sequence (list)
            Mathematically, the closer to 0.5 the better
        """
        n = len(X)
        
        Z_fake, fake_params = self.gen_noise(n, np.atleast_1d(self.Z_shape))
        Z_fake_res = self.transform(Z_fake)
        X_fake = np.c_[fake_params, Z_fake_res] if fake_params is not None else Z_fake_res
        self.dis_model.fit(X_fake, np.zeros((n,1)))
        self.dis_model.fit(X, y)
        
        Z_fake, fake_params = self.gen_noise(n, np.atleast_1d(self.Z_shape))
        Z_fake_res = self.gen_model.fit_forward(Z_fake)
        X_fake = np.c_[fake_params, Z_fake_res] if fake_params is not None else Z_fake_res
        y_predict = self.dis_model.fit_forward(X_fake)
        delta_dis = self.dis_model.back_propagation(y_predict, np.ones(y_predict.shape))
        if fake_params is not None:
            delta_dis = delta_dis[:,fake_params.shape[1]:]
        self.gen_model.back_propagation(y_predict=None, y=None, delta=delta_dis)
        self.gen_model.update_model()
        
        return (self.dis_model.transform(X).sum())
    
    def forward(self, Z):
        """Forward pass, simply call generative model to generate target with given data.
        
        Args:
            Z (np.array): input data
        
        Returns:
            Target generated
        """
        return self.gen_model.transform(Z)
