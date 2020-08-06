import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *
from gradiant_magnitude_approximation import GMA
from online_avg import OnlineAvg

from welford_var import Welford
#
# class TwoLayerNet(object):
#     """
#     A two-layer fully-connected neural network with ReLU nonlinearity and
#     softmax loss that uses a modular layer design. We assume an input dimension
#     of D, a hidden dimension of H, and perform classification over C classes.
#
#     The architecure should be affine - relu - affine - softmax.
#
#     Note that this class does not implement gradient descent; instead, it
#     will interact with a separate Solver object that is responsible for running
#     optimization.
#
#     The learnable parameters of the model are stored in the dictionary
#     self.params that maps parameter names to numpy arrays.
#     """
#     def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
#                  weight_scale=1e-3, reg=0.0):
#         """
#         Initialize a new network.
#
#         Inputs:
#         - input_dim: An integer giving the size of the input
#         - hidden_dim: An integer giving the size of the hidden layer
#         - num_classes: An integer giving the number of classes to classify
#         - weight_scale: Scalar giving the standard deviation for random
#           initialization of the weights.
#         - reg: Scalar giving L2 regularization strength.
#         """
#         self.params = {}
#         self.reg = reg
#
#         ############################################################################
#         # Initialize the weights and biases of the two-layer net. Weights          #
#         # should be initialized from a Gaussian centered at 0.0 with               #
#         # standard deviation equal to weight_scale, and biases should be           #
#         # initialized to zero. All weights and biases should be stored in the      #
#         # dictionary self.params, with first layer weights                         #
#         # and biases using the keys 'W1' and 'b1' and second layer                 #
#         # weights and biases using the keys 'W2' and 'b2'.                         #
#         ############################################################################
#         # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#
#         self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
#         self.params['b1'] = np.zeros((hidden_dim,))
#         self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
#         self.params['b2'] = np.zeros((num_classes,))
#
#         self.params['W1_var'] = np.ones((input_dim, hidden_dim)) #* self.reg
#         self.params['b1_var'] = np.ones((hidden_dim,)) #* self.reg
#         self.params['W2_var'] = np.ones((hidden_dim, num_classes)) #* self.reg
#         self.params['b2_var'] = np.ones((num_classes,)) #* self.reg
#         # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################
#
#     def __init__original(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
#                  weight_scale=1e-3, reg=0.0):
#         """
#         Initialize a new network.
#
#         Inputs:
#         - input_dim: An integer giving the size of the input
#         - hidden_dim: An integer giving the size of the hidden layer
#         - num_classes: An integer giving the number of classes to classify
#         - weight_scale: Scalar giving the standard deviation for random
#           initialization of the weights.
#         - reg: Scalar giving L2 regularization strength.
#         """
#         self.params = {}
#         self.reg = reg
#
#         ############################################################################
#         # Initialize the weights and biases of the two-layer net. Weights          #
#         # should be initialized from a Gaussian centered at 0.0 with               #
#         # standard deviation equal to weight_scale, and biases should be           #
#         # initialized to zero. All weights and biases should be stored in the      #
#         # dictionary self.params, with first layer weights                         #
#         # and biases using the keys 'W1' and 'b1' and second layer                 #
#         # weights and biases using the keys 'W2' and 'b2'.                         #
#         ############################################################################
#         # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#
#         self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
#         self.params['b1'] = np.zeros((hidden_dim,))
#         self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
#         self.params['b2'] = np.zeros((num_classes,))
#
#         # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################
#
#     def loss(self, X, y=None):
#         """
#         reg_per_weight
#         Compute loss and gradient for a minibatch of data.
#
#         Inputs:
#         - X: Array of input data of shape (N, d_1, ..., d_k)
#         - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
#
#         Returns:
#         If y is None, then run a test-time forward pass of the model and return:
#         - scores: Array of shape (N, C) giving classification scores, where
#           scores[i, c] is the classification score for X[i] and class c.
#
#         If y is not None, then run a training-time forward and backward pass and
#         return a tuple of:
#         - loss: Scalar value giving the loss
#         - grads: Dictionary with the same keys as self.params, mapping parameter
#           names to gradients of the loss with respect to those parameters.
#         """
#         scores = None
#         ############################################################################
#         # Implement the forward pass for the two-layer net, computing the          #
#         # class scores for X and storing them in the scores variable.              #
#         ############################################################################
#         # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#
#         w1 = self.params['W1']
#         w2 = self.params['W2']
#         b1 = self.params['b1']
#         b2 = self.params['b2']
#         mid, ar_cache = affine_relu_forward(X, w1, b1)
#         scores, a_cache = affine_forward(mid, w2, b2)
#
#         # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################
#
#         # If y is None then we are in test mode so just return scores
#         if y is None:
#             return scores
#
#         loss, grads = 0, {}
#         ############################################################################
#         # Implement the backward pass for the two-layer net. Store the loss        #
#         # in the loss variable and gradients in the grads dictionary. Compute data #
#         # loss using softmax, and make sure that grads[k] holds the gradients for  #
#         # self.params[k]. Don't forget to add L2 regularization!                   #
#         #                                                                          #
#         # NOTE: To ensure that your implementation matches ours and you pass the   #
#         # automated tests, make sure that your L2 regularization includes a factor #
#         # of 0.5 to simplify the expression for the gradient.                      #
#         ############################################################################
#         # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#
#         loss, dx = softmax_loss(scores, y)
#         loss += 0.5 * self.reg * np.sum(w1 * w1)# * self.param_var['w1']
#         loss += 0.5 * self.reg * np.sum(w2 * w2) #* self.param_var['W2']
#
#         dx, grads['W2'], grads['b2'] = affine_backward(dx, a_cache)
#
#         grads['W2'] += self.reg * w2
#
#         dx, grads['W1'], grads['b1'] = affine_relu_backward(dx, ar_cache)
#
#         grads['W1'] += self.reg * w1
#
#         # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################
#
#         return loss, grads
#
#     def loss_original(self, X, y=None):
#         """
#         Compute loss and gradient for a minibatch of data.
#
#         Inputs:
#         - X: Array of input data of shape (N, d_1, ..., d_k)
#         - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
#
#         Returns:
#         If y is None, then run a test-time forward pass of the model and return:
#         - scores: Array of shape (N, C) giving classification scores, where
#           scores[i, c] is the classification score for X[i] and class c.
#
#         If y is not None, then run a training-time forward and backward pass and
#         return a tuple of:
#         - loss: Scalar value giving the loss
#         - grads: Dictionary with the same keys as self.params, mapping parameter
#           names to gradients of the loss with respect to those parameters.
#         """
#         scores = None
#         ############################################################################
#         # Implement the forward pass for the two-layer net, computing the          #
#         # class scores for X and storing them in the scores variable.              #
#         ############################################################################
#         # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#
#         w1 = self.params['W1']
#         w2 = self.params['W2']
#         b1 = self.params['b1']
#         b2 = self.params['b2']
#         mid, ar_cache = affine_relu_forward(X, w1, b1)
#         scores, a_cache = affine_forward(mid, w2, b2)
#
#         # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################
#
#         # If y is None then we are in test mode so just return scores
#         if y is None:
#             return scores
#
#         loss, grads = 0, {}
#         ############################################################################
#         # Implement the backward pass for the two-layer net. Store the loss        #
#         # in the loss variable and gradients in the grads dictionary. Compute data #
#         # loss using softmax, and make sure that grads[k] holds the gradients for  #
#         # self.params[k]. Don't forget to add L2 regularization!                   #
#         #                                                                          #
#         # NOTE: To ensure that your implementation matches ours and you pass the   #
#         # automated tests, make sure that your L2 regularization includes a factor #
#         # of 0.5 to simplify the expression for the gradient.                      #
#         ############################################################################
#         # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#
#         loss, dx = softmax_loss(scores, y)
#         loss += 0.5 * self.reg * (np.sum(w1 * w1) + np.sum(w2 * w2))
#
#         dx, grads['W2'], grads['b2'] = affine_backward(dx, a_cache)
#
#         grads['W2'] += self.reg * w2
#
#         dx, grads['W1'], grads['b1'] = affine_relu_backward(dx, ar_cache)
#
#         grads['W1'] += self.reg * w1
#
#         # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
#         ############################################################################
#         #                             END OF YOUR CODE                             #
#         ############################################################################
#
#         return loss, grads
#

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None, iter_length=1000,
                 addaptive_reg=False,
                 divide_var_by_mean_var=True,
                 dropconnect=1, adaptive_dropconnect=False, var_normalizer=1,
                 variance_calculation_method='naive', static_variance_update=True,
                 inverse_var=True, adaptive_avg_reg=False, mean_mean=False):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropoWelfordut=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.adaptive_var_reg = addaptive_reg
        self.adaptive_avg_reg = False
        self.iter_length = iter_length
        self.divide_var_by_mean_var = divide_var_by_mean_var
        self.dropconnect = dropconnect
        self.adaptive_dropconnect = adaptive_dropconnect
        self.var_normalizer = var_normalizer
        self.variance_calculation_method = variance_calculation_method
        self.static_variance_update = static_variance_update
        self.dynamic_param_var = None
        self.inverse_var = inverse_var
        self.adaptive_avg_reg = adaptive_avg_reg
        self.mean_mean = mean_mean

        ############################################################################
        # Initialize the parameters of the network, storing all values in          #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        if self.adaptive_var_reg or self.adaptive_dropconnect:
            if self.variance_calculation_method == 'welford' or self.variance_calculation_method == 'GMA':
                self.dynamic_param_var = {}
            else:
                self.param_var = {}
                self.param_trajectories = {}

        if self.adaptive_avg_reg:
            self.param_avg = {}

        if self.num_layers == 1:
            dim = [input_dim, num_classes]
            self.params['W' + str(1)] = np.random.normal(scale=weight_scale, size=dim).astype(dtype)
            self.params['b' + str(1)] = np.zeros(dim[1], dtype=dtype)
            if self.adaptive_var_reg or self.adaptive_dropconnect:
                if self.variance_calculation_method == 'welford':
                    self.dynamic_param_var['W' + str(i + 1)] = \
                        Welford(dim=dim, var_normalizer=self.var_normalizer,
                                divide_var_by_mean_var=self.divide_var_by_mean_var,
                                static_var=self.static_variance_update)
                elif self.variance_calculation_method == 'GMA':
                    self.dynamic_param_var['W' + str(i + 1)] = \
                        GMA(dim=dim, var_normalizer=self.var_normalizer,
                            divide_var_by_mean_var=self.divide_var_by_mean_var,
                            static_var=self.static_variance_update)
                else:
                    self.param_var['W' + str(1)] = np.ones(shape=dim).astype(dtype)
                    self.param_trajectories['W' + str(1)] = []
            if self.adaptive_avg_reg:
                self.param_avg[f'W{i}'] = OnlineAvg(dim=dim, static_calculation=True)
        else:
            for i in range(self.num_layers):
                dim = []
                if i == 0:
                    dim.append(input_dim)
                    dim.append(hidden_dims[i])
                elif i == self.num_layers - 1:
                    dim.append(hidden_dims[i - 1])
                    dim.append(num_classes)
                else:
                    dim.append(hidden_dims[i - 1])
                    dim.append(hidden_dims[i])
                self.params['W' + str(i + 1)] = np.random.normal(scale=weight_scale, size=dim).astype(dtype)
                if self.adaptive_var_reg or self.adaptive_dropconnect:
                    if self.variance_calculation_method == 'welford':
                        self.dynamic_param_var['W' + str(i + 1)] =\
                            Welford(dim=dim, var_normalizer=self.var_normalizer,
                                    divide_var_by_mean_var=self.divide_var_by_mean_var,
                                    static_var=self.static_variance_update)
                    elif self.variance_calculation_method == 'GMA':
                        self.dynamic_param_var['W' + str(i + 1)] = \
                            GMA(dim=dim, var_normalizer=self.var_normalizer,
                                divide_var_by_mean_var=self.divide_var_by_mean_var,
                                static_var=self.static_variance_update)
                    else:
                        self.param_var['W' + str(i + 1)] = np.ones(shape=dim).astype(dtype)
                        self.param_trajectories['W' + str(i + 1)] = []
                # self.param_trajectories['W' + str(i + 1)] = np.zeros(shape=[self.iter_length] + dim).astype(dtype)
                self.params['b' + str(i + 1)] = np.zeros(dim[1], dtype=dtype)

                if self.adaptive_avg_reg:
                    self.param_avg[f'W{i}'] = OnlineAvg(dim=dim, static_calculation=True)

                if i != self.num_layers - 1 and (self.normalization == 'batchnorm' or self.normalization == 'layernorm'):
                    self.params['gamma' + str(i + 1)] = np.ones(dim[1], dtype=dtype)
                    self.params['beta' + str(i + 1)] = np.zeros(dim[1], dtype=dtype)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed
        if self.dropconnect != 1:
            self.dropconnect_param = {
                'mode': 'train',
                'p': dropconnect,
                'mc dropconnect forword passes': 10}
            if self.adaptive_dropconnect:
                if self.variance_calculation_method == 'naive':
                    self.dropconnect_param['adaptive_p'] = \
                        {param_name: param_var for param_name, param_var in self.param_var.items()}
                else:
                    self.dropconnect_param['adaptive_p'] = \
                        {param_name: self.dynamic_param_var[param_name].get_var() for param_name in
                         self.dynamic_param_var}
        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == 'batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization == 'layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.dropconnect != 1:
            self.dropconnect_param['mode'] = mode
        if self.normalization == 'batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        ############################################################################
        # Implement the forward pass for the fully-connected net, computing        #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        scores = X
        caches = []
        for i in range(self.num_layers):
            w = self.params['W' + str(i + 1)]
            b = self.params['b' + str(i + 1)]
            if i == self.num_layers - 1:
                scores, cache = affine_forward(scores, w, b)
            else:
                if self.normalization is None:
                    if self.dropconnect != 1 and i != self.num_layers -1:
                        scores, cache = affine_relu_dropconnect_forward(
                            scores, w, b, self.dropconnect_param,
                            adaptive_dropconnect_weight=self.dropconnect_param['adaptive_p'][f'W{i+1}'] if
                                self.adaptive_dropconnect else None)
                    else:
                        scores, cache = affine_relu_forward(scores, w, b)
                else:
                    gamma = self.params['gamma' + str(i + 1)]
                    beta = self.params['beta' + str(i + 1)]
                    if self.normalization == 'batchnorm':
                        scores, cache = affine_bn_relu_forward(scores, w, b, gamma, beta, self.bn_params[i])
                    elif self.normalization == 'layernorm':
                        scores, cache = affine_ln_relu_forward(scores, w, b, gamma, beta, self.bn_params[i])
                    else:
                        cache = None
            caches.append(cache)
            if self.use_dropout and i != self.num_layers - 1:
                scores, cache = dropout_forward(scores, self.dropout_param)
                caches.append(cache)

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # Implement the backward pass for the fully-connected net. Store the       #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        loss, dx = softmax_loss(scores, y)
        for i in range(self.num_layers):
            if self.adaptive_var_reg or self.adaptive_avg_reg:
                if self.adaptive_avg_reg:
                    reg_term = (self.params[w] - self.param_avg[w].get_static_mean()) ** 2
                    # reg_term = self.params[w] ** 2
                    if self.mean_mean:
                        reg_term /= np.mean(reg_term)
                else:
                    reg_term = self.params['W' + str(i + 1)] ** 2
                if self.adaptive_var_reg:
                    if self.variance_calculation_method in ['welford', 'GMA']:
                        var = self.dynamic_param_var[f'W{i+1}'].get_var()
                    else:
                        var = self.param_var[f'W{i+1}']
                    if not self.inverse_var:
                        var = 1/var
                    reg_term = reg_term.flatten() * var.flatten()
                # loss += 0.5 * self.reg * np.sum(w_sqr.flatten() * self.param_var[f'W{i+1}'].flatten())
                loss += 0.5 * self.reg * np.sum(reg_term)
            else:
                loss += 0.5 * self.reg * np.sum(self.params['W' + str(i + 1)] ** 2)
        for i in reversed(range(self.num_layers)):
            w = 'W' + str(i + 1)
            b = 'b' + str(i + 1)
            gamma = 'gamma' + str(i + 1)
            beta = 'beta' + str(i + 1)
            if i == self.num_layers - 1:
                dx, grads[w], grads[b] = affine_backward(dx, caches.pop())
            else:
                if self.use_dropout:
                    dx = dropout_backward(dx, caches.pop())
                if self.normalization is None:
                    # if self.use_dropconnect:
                    #     dx, grads[w], grads[b] = affine_relu_droconnect_backward(dx, caches.pop())
                    # else:
                    dx, grads[w], grads[b] = affine_relu_backward(dx, caches.pop())
                    
                #todo: implement dropconnect for the following
                elif self.normalization == 'batchnorm':
                    dx, grads[w], grads[b], grads[gamma], grads[beta] = affine_bn_relu_backward(dx, caches.pop())
                elif self.normalization == 'layernorm':
                    dx, grads[w], grads[b], grads[gamma], grads[beta] = affine_ln_relu_backward(dx, caches.pop())

            # if self.adaptive_var_reg or self.adaptive_avg_reg:
            if self.adaptive_avg_reg:
                reg_grad = self.params[w] - self.param_avg[w].get_static_mean()
            else:
                reg_grad = self.params[w]
            if self.adaptive_var_reg:
                if self.variance_calculation_method != 'naive':
                    var = self.dynamic_param_var[w].get_var()
                else:
                    var = self.param_var[w]
                if not self.inverse_var:
                    var = 1/var
                reg_grad = reg_grad.flatten()*var.flatten().reshape(self.params[w].shape)
            grads[w] += self.reg * (reg_grad)
                # grads[w] += self.reg * (self.params[w].flatten()*self.param_var[w].flatten()).reshape(
                #     self.params[w].shape)
            # else:
            #     grads[w] += self.reg * self.params[w]
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    #
    # def update_param_variances(self):
    #     for param_name, trajectory in self.param_trajectories.items():
    #         # trajectory = np.asarray(trajectory)
    #         var = np.var(trajectory, axis=0)
    #         print(f"avg unormalized param var: {np.avg(var)}")
    #         if self.divide_var_by_mean_var:
    #             var = var / np.avg(var)
    #             # self.param_var[param_name] = var / np.avg(var) #.var(trajectory, axis=0) / np.avg(trajectory)
    #         # else:
    #             self.param_var[param_name] = var
    #         if self.adaptive_dropconnect:
    #             droconnect_value = 1/2 + np.sqrt(1-4*var) / 2
    #             dropconnect_value = np.nan_to_num(droconnect_value, nan=0.5)
    #             self.dropconnect_param['adaptive_p'][param_name] = dropconnect_value
    #         # print(var)
    #         # print(f"avg param var: {np.avg(self.param_var[param_name])}")
    #     trajectory_names = self.param_trajectories.keys()
    #     for trajectory_name in trajectory_names:
    #         self.param_trajectories[trajectory_name] = []
