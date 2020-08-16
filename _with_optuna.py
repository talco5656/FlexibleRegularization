import functools

import pandas as pd

from allegroai import Task

# task = Task.get_task(project_name='Flexible Regularization', task_name='Simple CNN')
# As usual, a bit of setup
import matplotlib.pyplot as plt
from tabulate import tabulate

from cs231n.classifiers.cnn import *
from cs231n.classifiers.fc_net import FullyConnectedNet
from cs231n.classifiers.original_cnn import OriginalThreeLayerConvNet
from cs231n.classifiers.original_fc_net import FullyConnectedNetOriginal
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.adaptive_solver import AdaptiveSolver
from cs231n.solver import Solver

import argparse

import optuna


# task = Task.init(project_name='Flexible Regularization', task_name='Simple CNN')
task = Task.init(project_name='Flexible Regularization', task_name='Train and Eval single experiment')
# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# In[2]:
def representation(array, cycle=100):
    result_array = []
    for i in range(len(array)):
        if i % cycle == 0:
            result_array.append(array[i])
    return result_array


def parse_args():
    parser = argparse.ArgumentParser(description='Simple CNN')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--fc_width', type=int, default=200)
    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--verbose", type=int, default=False)
    parser.add_argument("--iter_length", type=int, default=500)
    # parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--model", default='mlp', choices=['mlp', 'cnn'])
    parser.add_argument("--num_trains", default=49000, type=int)
    parser.add_argument("--num_of_repeats", default=25, type=int)
    parser.add_argument("--dropconnect", default=1, type=float)
    parser.add_argument("--adaptive_var_reg", default=0, type=int)
    parser.add_argument("--adaptive_dropconnect", default=0, type=int)
    parser.add_argument("--divide_var_by_mean_var", default=0, type=int)
    parser.add_argument("--optimizer", default='sgd', choices=['sgd', 'sgd_momentum', 'adam', 'rmsprop'])
    return parser.parse_args()



def get_model(args, reg_strenght=0.1, iter_length=500, var_normalizer=1):
    if args.model == 'mlp':
        if args.adaptive_var_reg or args.adaptive_dropconnect or args.dropconnect != 1:
            model = FullyConnectedNet([args.fc_width] * 5, iter_length=iter_length, weight_scale=5e-2,
                                      reg=1 if args.divide_var_by_mean_var else reg_strenght, adaptive_reg=args.adaptive_var_reg,
                                      divide_var_by_mean_var=args.divide_var_by_mean_var,
                                      dropconnect=args.dropconnect, adaptive_dropconnect=args.adaptive_dropconnect,
                                      var_normalizer=1)
        else:
            model = FullyConnectedNetOriginal([args.fc_width]*5, weight_scale=5e-2, reg=reg_strenght)
    elif args.model == 'cnn':
        if args.addaptive_reg or args.addaptive_dropconnect or arg.dropconnect != 1:
            model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=args.fc_width, reg=reg_strenght, iter_length=args.iter_length)
        else:
            model = OriginalThreeLayerConvNet(weight_scale=0.001, hidden_dim=args.fc_width, reg=reg_strenght)
    return model


def train_and_eval_single_model(args):  #, trial):
    data = get_CIFAR10_data()
# def train_and_eval(data, model, solver, lr, reg_strength, num_trains):
#     original_params = {}
    num_train = min(args.num_trains, 49000) #[1000, 10000, 49000]
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }
    # learning_rates = [1e-3]  #[1e-3, 5e-3, 1e-4]

    # learning_rates = {'sgd': 5e-3, 'sgd_momentum': 1e-3, 'rmsprop': 1e-4, 'adam': 1e-3}
    optimizer = args.optimizer
    learning_rate = 5e-3  # trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    reg_strenght = 0.1  #trial.suggest_loguniform('reg_strength', 1e-15, 1)
    var_normalizer = 1  # trial.suggest_uniform('var_normalizer', 1, 1000)
    # reg_strenght = trial.suggest_loguniform('reg_strength', 1e-15, 1)
    batch_size = 100  # trial.suggest_int('bath_size', 50, 100, 25)
    iter_length = 1000 # trial.suggest_int('iter_length', 500, 1000, 250)
    model = get_model(args, reg_strenght=reg_strenght, iter_length=iter_length, var_normalizer=var_normalizer)
    # reg_strenghts = [1, 0.5, 0.1, 1e-2, 5e-3, 0]
    if args.adaptive_var_reg or args.adaptive_dropconnect or args.dropconnect != 1:
        solver = AdaptiveSolver(model, small_data, print_every=args.print_every,
                                     num_epochs=args.epochs, batch_size=batch_size,
                                     update_rule=optimizer,
                                     optim_config={
                                         'learning_rate': learning_rate,
                                     },
                                     verbose=args.verbose)
        solver.meta_train()
    else:
        solver = Solver(model, small_data, print_every=args.print_every,
                        num_epochs=args.epochs, batch_size=batch_size,
                        update_rule=args.optimizer,
                        optim_config={
                            'learning_rate': learning_rate
                        },
                        verbose=args.verbose)
        solver.train()
        #
    solvers = {}
    adaptive_solvers = {}
    result_dict = {}
    # for num_train in num_trains:
    #
    #     for reg_strenght in reg_strenghts:
    #         for update_rule in ['sgd', 'sgd_momentum', 'adam', 'rmsprop']:
    #             print(f'running adaptive with {update_rule}')
    #             model = get_models(args, reg_strenght)
    #             # adaptive_model = FullyConnectedNet([WIDTH]*5, iter_length=ITER_LENGTH, weight_scale=5e-2, reg=reg_strenght)
    #
    #
    #
    #             # adaptive_solver = AdaptiveSolver(model, small_data,
    #             #                                 print_every=500,
    #             #                                 num_epochs=EPOCHES, batch_size=100,
    #             #                                 update_rule=update_rule,
    #             #                                 optim_config={
    #             #                                 'learning_rate': lr,
    #             #                                       },
    #             #                               verbose=True)
    #             adaptive_solvers[update_rule] = adaptive_solver
    #             adaptive_solver.meta_train()
    #             print()
    #             print('Train result: train acc: %f; val_acc: %f' % (
    #                 adaptive_solver.best_train_acc, adaptive_solver.best_val_acc))
    #
    #
    #             print(f'running with {update_rule}')
    #             solver = Solver(original_model, small_data, print_every=args.print_every,
    #                             num_epochs=args.epochs, batch_size=args.batch_size,
    #                             update_rule=update_rule,
    #                             optim_config={
    #                                 'learning_rate': learning_rates[update_rule],
    #                             },
    #                             verbose=args.verbose)
    #             solvers[update_rule] = solver
    #             solver.train()
    #             print('Train result: train acc: %f; val_acc: %f' % (
    #                 solver.best_train_acc, solver.best_val_acc))
    #             print()
    #
    #
    #             # plt.subplot(3, 1, 1)
    #             # plt.title('Training loss')
    #             # plt.xlabel('Iteration')
    #             #
    #             # plt.subplot(3, 1, 2)
    #             # plt.title('Training accuracy')
    #             # plt.xlabel('Epoch')
    #             #
    #             # plt.subplot(3, 1, 3)
    #             # plt.title('Validation accuracy')
    #             # plt.xlabel('Epoch')
    #
    #             # for update_rule in ['sgd', 'sgd_momentum', 'adam', 'rmsprop']:
    #
    if args.verbose:
        plt.subplot(3, 1, 1)
        plt.title(f'Training loss. \n Reg: {reg_strenght}, Num trains: {num_train},')# LR: {lr}')
        plt.xlabel('Iteration')

        plt.subplot(3, 1, 2)
        plt.title('Training accuracy')
        plt.xlabel('Epoch')

        plt.subplot(3, 1, 3)
        plt.title('Validation accuracy')
        plt.xlabel('Epoch')


        # for update_rule, solver in solvers.items():
        plt.subplot(3, 1, 1)
        plt.plot(representation(solver.loss_history), 'o', label="loss_%s" % update_rule)

        # result_dict[(reg_strenghts, num_trains, update_rule, 'nonadaptive', 'loss_history')] = representation(solver.loss_history)

        plt.subplot(3, 1, 2)
        plt.plot(solver.train_acc_history, 'o', label="train_acc_%s" % update_rule)

        # result_dict[(reg_strenghts, num_trains, update_rule, 'nonadaptive', 'train_acc_history')] = solver.train_acc_history

        plt.subplot(3, 1, 3)

        plt.plot(solver.val_acc_history, 'o', label="val_acc_%s" % update_rule)

        # result_dict[(reg_strenght, num_train, update_rule, 'nonadaptive')] = solver
        #
        #     best_val_acc, best_experiment = np.max(solver.val_acc_history), \
        #                                     np.argmax(solver.val_acc_history)
            #     result_dict[(num_train, reg_strenght, update_rule, 'nonadaptive',
            #                  best_val_acc, solver.train_acc_history[best_experiment], solver.loss_history[best_experiment])] = solver
            # for update_rule, solver in adaptive_solvers.items():
            #     plt.subplot(3, 1, 1)
            #     plt.plot(representation(solver.loss_history), '-o', label="adaptive_loss_%s" % update_rule)
            #
            #     plt.subplot(3, 1, 2)
            #     plt.plot(solver.train_acc_history, '-o', label="adaptive_train_acc_%s" % update_rule)
            #
            #     # result_dict[(reg_strenghts, num_trains, update_rule, 'adaptive', 'train_acc_history')] = solver.train_acc_history
            #
            #     plt.subplot(3, 1, 3)
            #     plt.plot(solver.val_acc_history, '-o', label="adaptive_val_acc_%s" % update_rule)
            #     best_val_acc, best_experiment = np.max(solver.val_acc_history), \
            #                                     np.argmax(solver.val_acc_history)
            #     result_dict[(num_train, reg_strenght, update_rule, 'adaptive',
            #                  best_val_acc, solver.train_acc_history[best_experiment], solver.loss_history[best_experiment])] = solver
        for i in [1, 2, 3]:
            plt.subplot(3, 1, i)
            plt.legend(loc='upper center', ncol=4)
            plt.gcf().set_size_inches(15, 15)
        plt.show()
        # print(f"Best results, num train: {num_train}:")
        # # best_nonadaptive_descriotion = None
        # # best_nonadaptive_solver = None
        # # best_nonadaptive_val_acc = 0
        # #
        # # best_adaptive_descriotion = None
        # # best_adaptive_solver = None
        # # best_adaptive_val_acc = 0
        #
        # for desctiption, solver in result_dict.items():
        #     print(f"val acc history {solver.val_acc_history}")
        #     val_acc = np.max(solver.val_acc_history)  #[-1]
        #     print(f"{desctiption}, val_acc {val_acc}")
        #     if desctiption[3] == 'nonadaptive':
        #         if val_acc > best_nonadaptive_val_acc:
        #             best_nonadaptive_descriotion = desctiption
        #             best_nonadaptive_solver = solver
        #             best_nonadaptive_val_acc = val_acc
        #     else:
        #         if val_acc > best_adaptive_val_acc:
        #             # print(val_acc)
        #             best_adaptive_descriotion = desctiption
        #             best_adaptive_solver = solver
        #             best_adaptive_val_acc = val_acc
        # print("Best Nonadaptive solver:")
        # best_val_acc, best_experiment = np.max(best_nonadaptive_solver.val_acc_history), \
        #                                 np.argmax(best_nonadaptive_solver.val_acc_history)
        # print(f"Val acc: {best_nonadaptive_val_acc},"
        #       f" Train acc: {best_nonadaptive_solver.train_acc_history[best_experiment]},"
        #       f" loss: {best_nonadaptive_solver.loss_history[best_experiment]}")
        # print(best_nonadaptive_descriotion)
        # print("Best Adaptive solver:")
        # best_val_acc, best_experiment = np.max(best_adaptive_solver.val_acc_history), \
        #                                 np.argmax(best_adaptive_solver.val_acc_history)
        # print(f"Val acc: {best_adaptive_val_acc},"
        #       f" Train acc: {best_adaptive_solver.train_acc_history[best_experiment]},"
        #       f" loss: {best_adaptive_solver.loss_history[best_experiment]}")
        # print(best_adaptive_descriotion)
        # columns = ["Number of Trains", "Regularization",
        #            "Optimizer", "Adaptive?", "Validation acc", "Train acc", "Loss"]
        # table = pd.DataFrame(result_dict.keys(), columns=columns)
        # task.get_logger().report_table(title='Accuracy', series='Accuracy',
        #                    iteration=num_train, table_plot=table)
        # # print(table)
        #
        # print(tabulate(table, headers=columns))
        # return table
        # print()
    best_val_acc, best_experiment = np.max(solver.val_acc_history), \
                                    np.argmax(solver.val_acc_history)
    best_train_acc = solver.train_acc_history[best_experiment]
    print(best_val_acc, best_train_acc)
    return np.max(solver.val_acc_history)


def mean_and_ci_result(args):
    tables = []
    for _ in range(args.num_of_repeats):
        tables.append(train_and_eval(args))
    pd.concat(tables)
    content = [df.drop(columns=['Optimizer', 'Adaptive?']).values for df in tables]
    stacked_content = np.stack(content)
    mean_values = pd.DataFrame(np.avg(stacked_content, axis=0))
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
    task.get_logger().report_table(title='Mean values', series='Mean values',
                       iteration=args.num_trains, table_plot=mean_values)
    print("standard deviation")
    print(tabulate(std, headers=std.columns))
    task.get_logger().report_table(title='Standard deviation', series='Standard deviation',
                                   iteration=args.num_trains, table_plot=std)
    
    
if __name__ == "__main__":
    args = parse_args()
    # experiments = []
    train_and_eval_single_model(args)
    exit()
    objective = functools.partial(train_and_eval_single_model, args)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.num_of_repeats)
    # mean_and_ci_result(args)
    # train_and_eval(args)