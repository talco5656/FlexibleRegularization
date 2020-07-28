import numpy as np
import attr


# todo: does it work for matricies?
@attr.s
class GMA:
    """
    online variance calculation
    """
    dim = attr.ib()
    static_var = attr.ib(True)
    divide_var_by_mean_var = attr.ib(True)
    var_normalizer = attr.ib(1)
    beta = attr.ib(0.1)
    
    def __attrs_post_init__(self):
        self.count = 0
        # self.avg = np.zeros(self.dim)
        # self.M2 = np.zeros(self.dim)
        self.var = np.ones(self.dim)
        self.dynamic_var = np.ones(self.dim)
        
    def update(self, gradiant):
        self.count += 1
        self.dynamic_var = (1-self.beta)*self.dynamic_var + self.beta * gradiant * gradiant
    
    def get_beta(self):
        return self.beta ** self.count
    
    def update_var(self):
        self.var = self._get_var()
    
    def get_var(self):
        if self.static_var:
            return self.var
        return self._get_var()
    
    def _get_var(self):
        var = self.dynamic_var
        if self.divide_var_by_mean_var:
            var = var / np.avg(var)
        var = var * self.var_normalizer
        return var
