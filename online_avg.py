import numpy as np
import attr
import torch

@attr.s
class OnlineAvg:
    """
    online variance calculation
    For a new value newValue, compute the new count, new avg, the new M2.
    avg accumulates the avg of the entire dataset
    M2 aggregates the squared distance from the avg
    count aggregates the number of samples seen so far
    """
    dim = attr.ib()
    static_calculation = attr.ib(True)
    package = attr.ib('np')
    reinitiate_every_step = attr.ib(True)
    initial_param = attr.ib(None)

    def __attrs_post_init__(self):
        self.tensor_package = torch if self.package == 'torch' else np
        self.initial_param = self.tensor_package.clone(self.initial_param)
        self.count = 0
        self.avg = self.initial_param #or self.tensor_package.zeros(self.dim)
        self.static_avg = self.tensor_package.zeros(self.dim)

    def update(self, new_value):
        self.count += 1
        delta = new_value - self.avg
        self.avg += delta / self.count

    def update_static_mean(self):
        self.static_avg = self._get_avg()
        if self.reinitiate_every_step:
            self.count = 0
            self.avg = self.tensor_package.zeros(self.dim)
            self.static_avg = self.tensor_package.zeros(self.dim)

    def _get_avg(self):
        return self.avg

    def get_static_mean(self):
        if self.static_calculation:
            return self.static_avg
        return self._get_avg()

