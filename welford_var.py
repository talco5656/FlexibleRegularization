import numpy as np
import attr
import torch

@attr.s
class Welford:
    """
    online variance calculation
    For a new value newValue, compute the new count, new avg, the new M2.
    avg accumulates the avg of the entire dataset
    M2 aggregates the squared distance from the avg
    count aggregates the number of samples seen so far
    """
    dim = attr.ib()
    static_calculation = attr.ib(True)
    divide_var_by_mean_var = attr.ib(True)
    var_normalizer = attr.ib(1)
    # device = attr.ib('cpu')
    package = attr.ib('np')
    reinitiate_every_step = attr.ib(True)
    initial_param = attr.ib(None)

    def __attrs_post_init__(self):
        self.tensor_package = torch if self.package == 'torch' else np
        self.initial_param = self.tensor_package.clone(self.initial_param)
        self.count = 0
        self.mean = self.initial_param #if  or self.tensor_package.zeros(self.dim)
        self.M2 = self.tensor_package.zeros(self.dim)
        self.var = self.tensor_package.ones(self.dim)

        
    def update(self, new_value):
        self.count += 1
        delta = new_value - self.mean
        self.mean += delta / self.count  # todo: is this coordinate-wise? Yes, count is integer?
        delta2 = new_value - self.mean
        self.M2 += delta * delta  # todo: is this coordinate-wise? Yes it is.

    def update_var(self):
        self.var = self._get_var()
        if self.reinitiate_every_step:
            self.count = 0
            self.mean = self.tensor_package.zeros(self.dim)
            self.M2 = self.tensor_package.zeros(self.dim)
            self.var = self.tensor_package.ones(self.dim)

    def get_mean(self):
        return self.mean
    
    def get_mle_var(self):
        var = self.M2 / self.count - 1
        if self.divide_var_by_mean_var:
            var = var / self.tensor_package.mean(var)  # todo: is this coordinate-wise??
        var = var * self.var_normalizer
        return var


    def get_var(self):
        if self.static_calculation:
            return self.var
        return self._get_var()
    
    def _get_var(self):
        var = self.M2 / max((self.count - 1), 1)
        if self.divide_var_by_mean_var:
            var = var/self.tensor_package.mean(var)
        var = var * self.var_normalizer
        return var
