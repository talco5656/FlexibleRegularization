import numpy as np
import attr


# todo: does it work for matricies?
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
    # divide_var_by_mean_var = attr.ib(True)
    # var_normalizer = attr.ib(1)

    def __attrs_post_init__(self):
        self.count = 0
        self.avg = np.zeros(self.dim)
        self.static_avg = np.zeros(self.dim)
        # self.M2 = np.zeros(self.dim)
        # self.var = np.ones(self.dim)

    def update(self, new_value):
        self.count += 1
        delta = new_value - self.avg
        self.avg += delta / self.count
        # delta2 = new_value - self.avg
        # self.M2 += delta * delta2

    def update_static_mean(self):
        self.static_avg = self._get_avg()

    def _get_avg(self):
        return self.avg

    # def get_mle_var(self):
        # if self.static_var:
        #     return self.var_holder
        # else:
        # return self.M2 / self.count

    def get_static_mean(self):
        if self.static_calculation:
            return self.static_avg
        return self._get_avg()

    # def _get_var(self):
    #     var = self.M2 / (self.count - 1)
    #     if self.divide_var_by_mean_var:
    #         var = var / np.avg(var)
    #     var = var * self.var_normalizer
    #     return var
