#!/usr/bin/env python
# coding: utf-8
import argparse
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import attr
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

import torch.nn.functional as F  # useful stateless functions
from torchvision import models
from trains import Task
# In[2]:
import pytorch_addaptive_optim
import pandas as pd

DataTuple = namedtuple("DataTuple", ["loader_train", "loader_val", "loader_test"])


USE_GPU = True

dtype = torch.float32  # we will be using float throughout this tutorial



# print('using device:', device)



def flatten(x):
    N = x.shape[0]  # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


def test_flatten():
    x = torch.arange(12).view(2, 1, 3, 2)
    print('Before flattening: ', x)
    print('After flattening: ', flatten(x))


import torch.nn.functional as F  # useful stateless functions


def two_layer_fc(x, params):
    """
    A fully-connected neural networks; the architecture is:
    NN is fully connected -> ReLU -> fully connected layer.
    Note that this function only defines the forward pass;
    PyTorch will take care of the backward pass for us.

    The input to the network will be a minibatch of data, of shape
    (N, d1, ..., dM) where d1 * ... * dM = D. The hidden layer will have H units,
    and the output layer will produce scores for C classes.

    Inputs:
    - x: A PyTorch Tensor of shape (N, d1, ..., dM) giving a minibatch of
      input data.
    - params: A list [w1, w2] of PyTorch Tensors giving weights for the network;
      w1 has shape (D, H) and w2 has shape (H, C).

    Returns:
    - scores: A PyTorch Tensor of shape (N, C) giving classification scores for
      the input data x.
    """
    # first we flatten the image
    x = flatten(x)  # shape: [batch_size, C x H x W]

    w1, w2 = params
    x = F.relu(x.mm(w1))
    x = x.mm(w2)
    return x


def two_layer_fc_test():
    hidden_layer_size = 42
    x = torch.zeros((64, 50), dtype=dtype)  # minibatch size 64, feature dimension 50
    w1 = torch.zeros((50, hidden_layer_size), dtype=dtype)
    w2 = torch.zeros((hidden_layer_size, 10), dtype=dtype)
    scores = two_layer_fc(x, [w1, w2])
    print(scores.size())  # you should see [64, 10]



def three_layer_convnet(x, params):
    """
    Performs the forward pass of a three-layer convolutional network with the
    architecture defined above.

    Inputs:
    - x: A PyTorch Tensor of shape (N, 3, H, W) giving a minibatch of images
    - params: A list of PyTorch Tensors giving the weights and biases for the
      network; should contain the following:
      - conv_w1: PyTorch Tensor of shape (channel_1, 3, KH1, KW1) giving weights
        for the first convolutional layer
      - conv_b1: PyTorch Tensor of shape (channel_1,) giving biases for the first
        convolutional layer
      - conv_w2: PyTorch Tensor of shape (channel_2, channel_1, KH2, KW2) giving
        weights for the second convolutional layer
      - conv_b2: PyTorch Tensor of shape (channel_2,) giving biases for the second
        convolutional layer
      - fc_w: PyTorch Tensor giving weights for the fully-connected layer. Can you
        figure out what the shape should be?
      - fc_b: PyTorch Tensor giving biases for the fully-connected layer. Can you
        figure out what the shape should be?

    Returns:
    - scores: PyTorch Tensor of shape (N, C) giving classification scores for x
    """
    conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b = params
    scores = None

    scores = F.relu_(F.conv2d(x, conv_w1, conv_b1, padding=2))
    scores = F.relu_(F.conv2d(scores, conv_w2, conv_b2, padding=1))
    scores = F.linear(flatten(scores), fc_w.T, fc_b)

    return scores


def three_layer_convnet_test():
    x = torch.zeros((64, 3, 32, 32), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]

    conv_w1 = torch.zeros((6, 3, 5, 5), dtype=dtype)  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b1 = torch.zeros((6,))  # out_channel
    conv_w2 = torch.zeros((9, 6, 3, 3), dtype=dtype)  # [out_channel, in_channel, kernel_H, kernel_W]
    conv_b2 = torch.zeros((9,))  # out_channel

    # you must calculate the shape of the tensor after two conv layers, before the fully-connected layer
    fc_w = torch.zeros((9 * 32 * 32, 10))
    fc_b = torch.zeros(10)

    scores = three_layer_convnet(x, [conv_w1, conv_b1, conv_w2, conv_b2, fc_w, fc_b])
    print(scores.size())  # you should see [64, 10]


def random_weight(shape):
    """
    Create random Tensors for weights; setting requires_grad=True means that we
    want to compute gradients for these Tensors during the backward pass.
    We use Kaiming normalization: sqrt(2 / fan_in)
    """
    if len(shape) == 2:  # FC weight
        fan_in = shape[0]
    else:
        fan_in = np.prod(shape[1:])  # conv weight [out_channel, in_channel, kH, kW]
    # randn is standard normal distribution generator.
    w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2. / fan_in)
    w.requires_grad = True
    return w


def zero_weight(shape):
    return torch.zeros(shape, device=device, dtype=dtype, requires_grad=True)


def check_accuracy_part2(loader, model_fn, params):
    """
    Check the accuracy of a classification model.

    Inputs:
    - loader: A DataLoader for the data split we want to check
    - model_fn: A function that performs the forward pass of the model,
      with the signature scores = model_fn(x, params)
    - params: List of PyTorch Tensors giving parameters of the model

    Returns: Nothing, but prints the accuracy of the model
    """
    split = 'val' if loader.dataset.train else 'test'
    print('Checking accuracy on the %s set' % split)
    num_correct, num_samples = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.int64)
            scores = model_fn(x, params)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))


def train_part2(model_fn, params, learning_rate):
    """
    Train a model on CIFAR-10.

    Inputs:
    - model_fn: A Python function that performs the forward pass of the model.
      It should have the signature scores = model_fn(x, params) where x is a
      PyTorch Tensor of image data, params is a list of PyTorch Tensors giving
      model weights, and scores is a PyTorch Tensor of shape (N, C) giving
      scores for the elements in x.
    - params: List of PyTorch Tensors giving weights for the model
    - learning_rate: Python scalar giving the learning rate to use for SGD

    Returns: Nothing
    """
    for t, (x, y) in enumerate(loader_train):
        # Move the data to the proper device (GPU or CPU)
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=torch.long)

        # Forward pass: compute scores and loss
        scores = model_fn(x, params)
        loss = F.cross_entropy(scores, y)

        # Backward pass: PyTorch figures out which Tensors in the computational
        # graph has requires_grad=True and uses backpropagation to compute the
        # gradient of the loss with respect to these Tensors, and stores the
        # gradients in the .grad attribute of each Tensor.
        loss.backward()

        # Update parameters. We don't want to backpropagate through the
        # parameter updates, so we scope the updates under a torch.no_grad()
        # context manager to prevent a computational graph from being built.
        with torch.no_grad():
            for w in params:
                w -= learning_rate * w.grad

                # Manually zero the gradients after running the backward pass
                w.grad.zero_()

        if t % print_every == 0:
            print('Iteration %d, loss = %.4f' % (t, loss.item()))
            check_accuracy_part2(loader_val, model_fn, params)
            print()


class TwoLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        # assign layer objects to class attributes
        self.fc1 = nn.Linear(input_size, hidden_size)
        # nn.init package contains convenient initialization methods
        # http://pytorch.org/docs/master/nn.html#torch-nn-init
        nn.init.kaiming_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        # forward always defines connectivity
        x = flatten(x)
        scores = self.fc2(F.relu(self.fc1(x)))
        return scores


def test_TwoLayerFC():
    input_size = 50
    x = torch.zeros((64, input_size), dtype=dtype)  # minibatch size 64, feature dimension 50
    model = TwoLayerFC(input_size, 42, 10)
    scores = model(x)
    print(scores.size())  # you should see [64, 10]


class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, channel_1, 5, padding=2)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(channel_1, channel_2, 3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.fc = nn.Linear(channel_2 * 32 * 32, num_classes)
        nn.init.kaiming_normal_(self.fc.weight)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        scores = None

        scores = self.relu(self.conv1(x))
        scores = self.relu(self.conv2(scores))
        scores = self.fc(flatten(scores))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return scores


def test_ThreeLayerConvNet():
    x = torch.zeros((64, 3, 32, 32), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]
    model = ThreeLayerConvNet(in_channel=3, channel_1=12, channel_2=8, num_classes=10)
    scores = model(x)
    print(scores.size())  # you should see [64, 10]


class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            self.relu,
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            self.relu,
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            self.relu,

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            self.relu,

        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x: Tensor = self.avgpool(x)
        x = x.view(-1, 7 * 7 * 256)
        x = self.classifier(x)
        return x


def test1():
    test_flatten()

    two_layer_fc_test()

    three_layer_convnet_test()

    random_weight((3, 5))

    hidden_layer_size = 4000
    learning_rate = 1e-2
    model = TwoLayerFC(3 * 32 * 32, hidden_layer_size, 10)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.1)

    train_part34(model, optimizer)


    learning_rate = 3e-3
    channel_1 = 32
    channel_2 = 16

    model = None
    optimizer = None
    ################################################################################
    # TODO: Instantiate your ThreeLayerConvNet model and a corresponding optimizer #
    ################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    model = ThreeLayerConvNet(3, channel_1, channel_2, 10)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_part34(model, optimizer)


def test_seq():
    hidden_layer_size = 4000
    learning_rate = 5e-4
    epochs = 10

    regular_model = nn.Sequential(
        Flatten(),
        nn.Linear(3 * 32 * 32, hidden_layer_size),
        nn.ReLU(),
        nn.Linear(hidden_layer_size, 10),
    )
    adaptive_model = nn.Sequential(
        Flatten(),
        nn.Linear(3 * 32 * 32, hidden_layer_size),
        nn.ReLU(),
        nn.Linear(hidden_layer_size, 10),
    )
    # you can use Nesterov momentum in optim.SGD
    optimizer = optim.SGD(regular_model.parameters(), lr=learning_rate,
                          momentum=1, nesterov=False)
    adaptive_optimizer = optim.SGD(adaptive_model.parameters(), lr=learning_rate,
                                   momentum=1, nesterov=False, weight_decay=0.01)
    # adaptive_optimizer = pytorch_addaptive_optim.sgd.SGD(adaptive_model.parameters(), lr=learning_rate,
    #                       momentum=0.9, nesterov=True, weight_decay=0.1, adaptive_weight_decay=True, iter_length=200)

    print("regular model:")
    train_part34(regular_model, optimizer, epochs=epochs)
    # print("adaptive model:")
    print("l2 model:")
    train_part34(adaptive_model, adaptive_optimizer, epochs=epochs)


def test_conv_seq():
    channel_1 = 32
    channel_2 = 16
    learning_rate = 1e-2

    model = nn.Sequential(
        nn.Conv2d(3, channel_1, 5, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(channel_1, channel_2, 3, padding=1),
        nn.ReLU(inplace=True),
        Flatten(),
        nn.Linear(channel_2 * 32 * 32, 10)
    )
    # for i in (0,2,5):
    #     w_shape=model[i].weight.data.shape
    #     b_shape=model[i].bias.data.shape
    #     model[i].weight.data=random_weight(w_shape)
    #     model[i].bias.data=zero_weight(b_shape)

    optimizer = optim.SGD(model.parameters(), nesterov=False, lr=learning_rate, momentum=1)
    print("regular model:")
    train_part34(model, optimizer, epochs=5)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ################################################################################
    #                                 END OF YOUR CODE
    ################################################################################

    # train_part34(model, optimizer)
    # optimizer = optim.SGD(regular_model.parameters(), lr=learning_rate,
    #                       momentum=0.9, nesterov=True)
    model = nn.Sequential(
        nn.Conv2d(3, channel_1, 5, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(channel_1, channel_2, 3, padding=1),
        nn.ReLU(inplace=True),
        Flatten(),
        nn.Linear(channel_2 * 32 * 32, 10)
    )
    # optimizer = pytorch_addaptive_optim.sgd.SGD(model.parameters(), lr=learning_rate,
    #                       momentum=0.9, nesterov=True, weight_decay=0.1, adaptive_weight_decay=True, iter_length=200)
    #
    optimizer = optim.SGD(model.parameters(), nesterov=False, lr=learning_rate, momentum=1, weight_decay=0.01)
    # print("regular model:")
    # train_part34(model, optimizer)
    print("with l2 0.01:")
    train_part34(model, optimizer, epochs=5)

    model = nn.Sequential(
        nn.Conv2d(3, channel_1, 5, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(channel_1, channel_2, 3, padding=1),
        nn.ReLU(inplace=True),
        Flatten(),
        nn.Linear(channel_2 * 32 * 32, 10)
    )
    # optimizer = pytorch_addaptive_optim.sgd.SGD(model.parameters(), lr=learning_rate,
    #                       momentum=0.9, nesterov=True, weight_decay=0.1, adaptive_weight_decay=True, iter_length=200)
    #
    optimizer = optim.SGD(model.parameters(), nesterov=False, lr=learning_rate, momentum=1, weight_decay=0.1)
    # print("regular model:")
    # train_part34(model, optimizer)
    print("with l2 0.1:")
    train_part34(model, optimizer, epochs=5)


def test_alexnet():
    learning_rate = 1e-3
    model = AlexNet()
    # optimizer = optim.Adam(model.parameters())
    optimizer = pytorch_addaptive_optim.sgd.SGD(model.parameters(), lr=learning_rate,
                                                momentum=0.9, nesterov=True, weight_decay=0.1,
                                                adaptive_weight_decay=True)
    train_part34(model, optimizer, epochs=2)
    best_model = model
    check_accuracy_part34(loader_test, best_model)


def parse_args():
    parser = argparse.ArgumentParser(description='Simple CNN')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--fc_width', type=int, default=200)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--iter_length", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--model", default='mlp') #, choices=['mlp', 'cnn', 'alexnet',])
    parser.add_argument("--num_trains", default=49000, type=int)
    parser.add_argument("--num_of_repeats", default=1, type=int)
    parser.add_argument("--dropconnect", default=1, type=float)
    parser.add_argument("--adaptive_var_reg", default=1, type=int)
    parser.add_argument("--reg_strength", default=None, type=float)
    parser.add_argument("--adaptive_dropconnect", default=0, type=int)
    parser.add_argument("--divide_var_by_mean_var", default=1, type=int)
    parser.add_argument("--test", default=0, type=int)
    parser.add_argument("--variance_calculation_method", default="welford", choices=["naive", "welford", "GMA"])
    parser.add_argument("--static_variance_update", default=1, type=int)
    parser.add_argument("--var_normalizer", default=1, type=float)  # todo: make sure this is the right value to put
    parser.add_argument("--batchnorm", default=0, type=int, help="Available only for MLP.")
    parser.add_argument("--optimizer", default='sgd', choices=['sgd', 'sgd_momentum', 'adam', 'rmsprop', None])
    parser.add_argument("--baseline_as_well", default=1, type=int)
    parser.add_argument("--eval_distribution_sample", default=0, type=float)
    parser.add_argument("--inverse_var", default=1, type=int)
    parser.add_argument("--adaptive_avg_reg", default=0, type=int)
    parser.add_argument("--mean_mean", default=0, type=int)
    parser.add_argument("--trains", default=1, type=int)
    parser.add_argument("--hidden_layers", default=5, type=int)
    parser.add_argument("--lnn", default=0, type=int)
    parser.add_argument("--reg_layers", default='1,2,3')
    parser.add_argument("--momentum", type=int, default=0)
    parser.add_argument("--nesterov", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
    return parser.parse_args()



@attr.s
class TorchExample():
    args = attr.ib()

    def __attrs_post_init__(self):
        self.num_trains = min(self.args.num_trains, 49000)
        if self.args.dataset == 'cifar10':
            self.data, self.num_classes = self.get_pytorch_cifar_data()
        elif self.args.dataset == 'cifar 100':
            self.data, self.num_classes = self.get_pytorch_cifar100_data()
        if self.args.gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.logger = Task.current_task().get_logger() if self.trains else None

    def get_pytorch_imagenet_data(self):
        #todo: change hard wired arguments
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])
        cifar10_train = dset.imagenet('./cs231n/datasets', train=True, download=True,
                                      transform=transform)
        loader_train = DataLoader(cifar10_train, batch_size=self.args.batch_size,
                                  sampler=sampler.SubsetRandomSampler(range(self.num_trains)))

        cifar10_val = dset.imagenet('./cs231n/datasets', train=True, download=True,
                                   transform=transform)
        loader_val = DataLoader(cifar10_val, batch_size=self.args.batch_size,
                                sampler=sampler.SubsetRandomSampler(range(self.num_trains, self.num_trains + 1000)))

        cifar10_test = dset.imagenet('./cs231n/datasets', train=False, download=True,
                                    transform=transform)
        loader_test = DataLoader(cifar10_test, batch_size=self.args.batch_size)
        return DataTuple(loader_train, loader_val, loader_test)

    def get_pytorch_cifar100_data(self):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        cifar100_train = dset.CIFAR100('./cs231n/datasets', train=True, download=True,
                                     transform=transform)
        loader_train = DataLoader(cifar100_train, batch_size=self.args.batch_size,
                                  sampler=sampler.SubsetRandomSampler(range(self.num_trains)))

        cifar100_val = dset.CIFAR100('./cs231n/datasets', train=True, download=True,
                                   transform=transform)
        loader_val = DataLoader(cifar100_val, batch_size=self.args.batch_size,
                                sampler=sampler.SubsetRandomSampler(range(self.num_trains, self.num_trains + 1000)))

        cifar100_test = dset.CIFAR100('./cs231n/datasets', train=False, download=True,
                                    transform=transform)
        loader_test = DataLoader(cifar100_test, batch_size=self.args.batch_size)
        return DataTuple(loader_train, loader_val, loader_test), 100

    def get_pytorch_cifar_data(self):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                                     transform=transform)
        loader_train = DataLoader(cifar10_train, batch_size=self.args.batch_size,
                                  sampler=sampler.SubsetRandomSampler(range(self.num_trains)))

        cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                                   transform=transform)
        loader_val = DataLoader(cifar10_val, batch_size=self.args.batch_size,
                                sampler=sampler.SubsetRandomSampler(range(self.num_trains, self.num_trains + 1000)))

        cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True,
                                    transform=transform)
        loader_test = DataLoader(cifar10_test, batch_size=self.args.batch_size)
        return DataTuple(loader_train, loader_val, loader_test), 10

    def get_mlp_model(self):
        hidden_layer_size = 4000
        return nn.Sequential(
            Flatten(),
            nn.Linear(3 * 32 * 32, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, self.num_classes),
        )

    def get_cnn_model(self):
        channel_1 = 32
        channel_2 = 16
        return nn.Sequential(
            nn.Conv2d(3, channel_1, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_1, channel_2, 3, padding=1),
            nn.ReLU(inplace=True),
            Flatten(),
            nn.Linear(channel_2 * 32 * 32, self.num_classes)
        )

    def check_accuracy(self, loader, model):
        if loader.dataset.train:
            print('Checking accuracy on validation set')
        else:
            print('Checking accuracy on test set')
        num_correct = 0
        num_samples = 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=self.device, dtype=dtype)  # move to device, e.g. GPU
                y = y.to(device=self.device, dtype=torch.long)
                scores = model(x)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
            return num_correct, num_samples, acc

    def general_train(self, model, optimizer, epochs=1):
        """
        Train a model on CIFAR-10 using the PyTorch Module API.

        Inputs:
        - model: A PyTorch Module giving the model to train.
        - optimizer: An Optimizer object we will use to train the model
        - epochs: (Optional) A Python integer giving the number of epochs to train for

        Returns: Nothing, but prints model accuracies during training.
        """
        model = model.to(device=self.device)  # move the model parameters to CPU/GPU
        best_val_acc, reported_train_acc, best_iteration = 0, 0, 0
        val_loader = self.data.loader_test if self.test else self.data.loader_val
        for e in range(epochs):
            for t, (x, y) in enumerate(self.data.loader_train):
                model.train()  # put model to training mode
                x = x.to(device=self.device, dtype=dtype)  # move to device, e.g. GPU
                y = y.to(device=self.device, dtype=torch.long)

                scores = model(x)
                loss = F.cross_entropy(scores, y)
                # Zero out all of the gradients for the variables which the optimizer
                # will update.
                optimizer.zero_grad()

                # This is the backwards pass: compute the gradient of the loss with
                # respect to each  parameter of the model.
                loss.backward()

                # Actually update the parameters of the model using the gradients
                # computed by the backwards pass.
                optimizer.step()

                if t % self.args.print_every == 0:
                    print('Iteration %d, loss = %.4f' % (t, loss.item()))
                    num_correct, num_samples, val_acc = self.check_accuracy(val_loader, model)
                    _, _, train_acc = self.check_accuracy(self.data.loader_train, model)
                    if self.logger:
                        self.logger.report_scalar(value=train_acc, title='Train accuracy', series='solver', iteration=t)
                        self.logger.report_scalar(value=val_acc, title='Val accuracy', series='solver', iteration=t)
                    if best_val_acc < val_acc:
                        best_val_acc, reported_train_acc, best_iteration = \
                            val_acc, train_acc, t + e * self.num_trains // self.args.batch_size
        return best_val_acc, reported_train_acc, best_iteration

    def get_model(self, reg_layers):
        if self.args.model == 'mlp':
            return self.get_mlp_model()
        if self.args.model == 'cnn':
            return self.get_cnn_model()
        if self.args.model == 'alexnet':
            return AlexNet()
        if self.args.model == "resnet18":
            model = models.resnet18(pretrained=self.args.pretrained)
            model.fc = nn.Linear(512, self.num_classes)
            return model
        if self.args.model == "resnet50":
            model = models.resnet50(pretrained=self.args.pretrained)
            model.fc = nn.Linear(512, self.num_classes)
            return model

    def train_and_eval(self):

        learning_rates = {'sgd': 5e-3, 'sgd_momentum': 1e-3, 'rmsprop': 1e-4, 'adam': 1e-3}
        if isinstance(self.args.reg_strength, float):
            reg_strenghts = [self.args.reg_strength]
        else:
            reg_strenghts = [1, 0.1, 0.01, 0]
        if self.args.optimizer:
            update_rules = [self.args.optimizer]
        else:
            update_rules = ['sgd', 'sgd_momentum', 'adam', 'rmsprop']
        solvers = {}
        adaptive_solvers = {}
        result_dict = {}
        reg_layers = self.args.reg_layers.split(',') if self.args.reg_layers else ['1', '2', '3']
        for reg_strenght in reg_strenghts:
            # original_model = self.get_model(reg_layers)
            # original_optimizer = optim.SGD(original_model.parameters(), nesterov=self.args.nesterov,
            #                      lr=self.args.lr, momentum=self.args.momentum,
            #                                weight_decay=reg_strenght)

            # result_dict["Regular model"] = self.general_train(original_model, original_optimizer, epochs=self.args.epochs)

            adaptive_model = self.get_model(reg_layers)
            adaptive_optimizer = pytorch_addaptive_optim.sgd.SGD(adaptive_model.parameters(), lr=self.args.lr,
                                                                momentum=self.args.momentum, nesterov=self.args.nesterov,
                                                                 weight_decay=reg_strenght, adaptive_weight_decay=True, iter_length=200,
                                                                 device=self.device, logger=self.logger)
            result_dict["Adaptive model"] = self.general_train(adaptive_model, adaptive_optimizer, epochs=self.args.epochs)
        result_df = pd.DataFrame(result_dict, index=["Val acc", "Train acc", "iteration"]).transpose()

        return result_df

    def mean_and_ci_result(self):
        if self.args.trains:
            task = Task.init(project_name='Flexible Regularization',
                             task_name='Torch Models')  # , reuse_last_task_id=False)
        else:
            task = None
        tables = []
        for repeat_index in range(self.args.num_of_repeats):
            result_df = self.train_and_eval()
            tables.append(result_df)
            if self.args.trains:
                self.logger.report_table(title="Accuracy", series="Accuracy",
                                                    iteration=repeat_index, table_plot=result_df)
        # tables = pd.concat(tables)
        mean_values = np.mean(tables, axis=0)
        mean_values = pd.DataFrame(mean_values, index=["Regular model", "Adaptive model"],
                                   columns=["Val acc", "Train acc", "Iteratoin"])
        print(tabulate(mean_values, headers=mean_values.columns))
        if self.args.trains:
            self.logger.report_table(title="Accuracy", series="Accuracy",
                                           iteration=self.args.num_trains, table_plot=mean_values)
        exit()
        content = [df.drop(columns=['Optimizer', 'Adaptive?']).values for df in tables]
        stacked_content = np.stack(content)
        mean_values = pd.DataFrame(np.mean(stacked_content, axis=0))
        std = pd.DataFrame(np.std(stacked_content, axis=0))
        print(mean_values)
        print(std)
        second_column, third_column = tables[0]['Optimizer'], tables[0]['Adaptive?']
        mean_values.insert(loc=2, column='Optimizer', value=second_column)
        mean_values.insert(loc=3, column='Adaptive', value=third_column)
        mean_values.columns = tables[0].columns
        std.insert(loc=2, column='Optimizer', value=second_column)
        std.insert(loc=3, column='Adaptive', value=third_column)
        std.columns = tables[0].columns
        print("avg values")
        print(tabulate(mean_values, headers=mean_values.columns))
        if self.args.trains:
            task.get_logger().report_table(title='Mean values', series='Mean values',
                                           iteration=self.args.num_trains, table_plot=mean_values)
        print("standard deviation")
        print(tabulate(std, headers=std.columns))
        if self.args.trains:
            task.get_logger().report_table(title='Standard deviation', series='Standard deviation',
                                           iteration=self.args.num_trains, table_plot=std)

def main():
    args = parse_args()
    # mean_and_ci_result(args)
    torch_example = TorchExample(args)
    # torch_example.train_and_eval()
    torch_example.mean_and_ci_result()


if __name__ == "__main__":
    main()

    # test_seq()
    # test_conv_seq()
    # test_alexnet()