import torch
import trains
from torch.optim.optimizer import Optimizer, required

from welford_var import Welford

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, adaptive_weight_decay=False, iter_length=100, device=device,
                 noninverse_var=False, adaptive_avg_reg=False, logger=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)
        if adaptive_weight_decay:
            self.online_param_var_dict = self.create_online_param_var_dict()
            self.num_of_steps = 0
            self.iter_length = iter_length
            self.device = device
            self.noninverse_var = noninverse_var,
            self.logger = logger
        else:
            self.online_param_var_dict = None
        # if adaptive_avg_reg:
        #     self.avg_dict =
        # else:
        #     self.avg_dict = False

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def create_online_param_var_dict(self):
        # todo: implement in Pytorch instead numpy
        import numpy as np
        dtype = np.float32
        online_param_var = {}
        for group_index, param_group in enumerate(self.param_groups):
            for param_index, param in enumerate(param_group['params']):
                param_name = param.name
                if not param_name:
                    param_name = (group_index, param_index)
                # self.params[param_name] = param#.astype(dtype)
                # if self.adaptive_var_reg and 'W' in k:  # or (self.adaptive_dropconnect and k in ('W1', 'W2')):
                    # if self.variance_calculation_method == 'welford':
                online_param_var[param_name] = Welford(dim=param.shape, package='torch')
        return online_param_var

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_index, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for parameter_index, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    if self.online_param_var_dict:
                        parameter_name = (group_index, parameter_index)
                        var_tensor = self.online_param_var_dict[parameter_name].get_var().to(device=self.device)
                        if self.noninverse_var:
                            var_tensor = torch.inverse(var_tensor)
                        reg_p = p.mul(var_tensor)
                        d_p = d_p.add(reg_p, alpha=weight_decay)
                    else:
                        d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group['lr'])
                if self.online_param_var_dict and weight_decay != 0:
                    self.online_param_var_dict[parameter_name].update(p.to(device='cpu'))
                    if self.num_of_steps > 0 and self.num_of_steps % self.iter_length == 0:
                        self.online_param_var_dict[parameter_name].update_var()
                        # report var
                        if self.logger:
                        # logger = trains.Task.current_task().get_logger()
                            var_calculator = self.model.online_param_var[parameter_name]
                            d_var = var_calculator.M2 / (var_calculator.count - 1)
                            self.logger.report_scalar(
                                title=f"parameter variance, {self.model.reg}", series=parameter_name,
                                value=torch.average(d_var), iteration=self.num_of_steps)

        self.num_of_steps += 1
        return loss

    def update_param_variance_online(self, iteration):
        if not self.model.static_variance_update:
            return
        # logger = trains.Task.current_task().get_logger()
        for param_name in self.online_param_var:
            self.model.online_param_var[param_name].update_var()
            # var_calculator = self.model.online_param_var[param_name]
            # d_var = var_calculator.dynamic_var if \
            # self.model.variance_calculation_method == 'GMA' \
            # else var_calculator.M2 / (var_calculator.count - 1)
            # logger.report_scalar(
            #     title=f"parameter variance, {self.model.reg}", series=param_name, value=np.average(d_var), iteration=iteration)
            # if self.model.adaptive_dropconnect:
            #     var = self.model.online_param_var[param_name].get_var()
            #     droconnect_value = 1/2 + np.sqrt(1-4*var) / 2
            #     dropconnect_value = np.nan_to_num(droconnect_value, nan=0.5)
            #     if self.model.divide_var_by_mean_var:
            #         dropconnect_value = dropconnect_value / np.mean(dropconnect_value)
            #     dropconnect_value = dropconnect_value * self.model.dropconnect
            #     self.model.dropconnect_param['adaptive_p'][param_name] = dropconnect_value

