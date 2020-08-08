import pandas as pd

from trains import Task

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


# task = Task.init(project_name='Flexible Regularization', task_name='Simple CNN')
# task = Task.init(project_name='Flexible Regularization', task_name='Train and Eval')
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
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--iter_length", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--model", default='mlp', choices=['mlp', 'cnn'])
    parser.add_argument("--num_trains", default=49000, type=int)
    parser.add_argument("--num_of_repeats", default=10, type=int)
    parser.add_argument("--dropconnect", default=1, type=float)
    parser.add_argument("--adaptive_var_reg", default=0, type=int)
    parser.add_argument("--reg_strength", default=None, type=float)
    parser.add_argument("--adaptive_dropconnect", default=0, type=int)
    parser.add_argument("--divide_var_by_mean_var", default=0, type=int)
    parser.add_argument("--test", default=0, type=int)
    parser.add_argument("--variance_calculation_method", default="naive", choices=["naive", "welford", "GMA"])
    parser.add_argument("--static_variance_update", default=1, type=int)
    parser.add_argument("--var_normalizer", default=1, type=float)
    parser.add_argument("--batchnorm", default=0, type=int, help="Available only for MLP.")
    parser.add_argument("--optimizer", default=None, choices=['sgd', 'sgd_momentum', 'adam', 'rmsprop', None])
    parser.add_argument("--baseline_as_well", default=1, type=int)
    parser.add_argument("--eval_distribution_sample", default=0, type=float)
    parser.add_argument("--inverse_var", default=1, type=int)
    parser.add_argument("--adaptive_avg_reg", default=0, type=int)
    parser.add_argument("--mean_mean", default=0, type=int)
    parser.add_argument("--trains", default=1, type=int)
    parser.add_argument("--hidden_layers", default=5, type=int)
    parser.add_argument("--lnn", default=0, type=int)
    return parser.parse_args()


def get_models(args, reg_strenght=0.1):
    if args.model == 'mlp':
        original_model = FullyConnectedNetOriginal([args.fc_width]*args.hidden_layers, weight_scale=5e-2, reg=reg_strenght,
                                                   normalization="batchnorm" if args.batchnorm else None)
        adaptive_model = FullyConnectedNet([args.fc_width] * args.hidden_layers, normalization="batchnorm" if args.batchnorm else None,
                                           iter_length=args.iter_length, weight_scale=5e-2,
                                           reg=1 if not args.divide_var_by_mean_var else reg_strenght,
                                           addaptive_reg=args.adaptive_var_reg,
                                           divide_var_by_mean_var=args.divide_var_by_mean_var,
                                           dropconnect=args.dropconnect, adaptive_dropconnect=args.adaptive_dropconnect,
                                           static_variance_update=args.static_variance_update,
                                           variance_calculation_method=args.variance_calculation_method,
                                           var_normalizer=args.var_normalizer,
                                           inverse_var=args.inverse_var,
                                           adaptive_avg_reg=args.adaptive_avg_reg,
                                           mean_mean=args.mean_mean,
                                           lnn=args.lnn)
    elif args.model == 'cnn':
        original_model = OriginalThreeLayerConvNet(weight_scale=0.001, hidden_dim=args.fc_width, reg=reg_strenght)
        adaptive_model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=args.fc_width, reg=reg_strenght,
                                           iter_length=args.iter_length,
                                           variance_calculation_method=args.variance_calculation_method,
                                           dropconnect=args.dropconnect,
                                           static_variance_update=args.static_variance_update,
                                           adaptive_dropconnect=args.adaptive_dropconnect,
                                           var_normalizer=args.var_normalizer,
                                           inverse_var=args.inverse_var,
                                           adaptive_avg_reg=args.adaptive_avg_reg,
                                           mean_mean=args.mean_mean)
    return original_model, adaptive_model


def train_and_eval(args):
    if args.trains:
        task = Task.current_task()
    data = get_CIFAR10_data()
    num_train = min(args.num_trains, 49000)
    learning_rates = {'sgd': 5e-3, 'sgd_momentum': 1e-3, 'rmsprop': 1e-4, 'adam': 1e-3}
    if isinstance(args.reg_strength, float):
        reg_strenghts = [args.reg_strength]
    else:
        reg_strenghts = [1, 0.5, 0.1, 1e-2, 5e-3, 0]
    if args.optimizer:
        update_rules = [args.optimizer]
    else:
        update_rules = ['sgd', 'sgd_momentum', 'adam', 'rmsprop']
    solvers = {}
    print("update rules: ", update_rules)
    print("args.optimizer", args.optimizer)
    print("args.reg_strengths", args.reg_strengths)
    print("reg strengths: ", reg_strenghts)
    adaptive_solvers = {}
    result_dict = {}
    if args.test:
        small_data = {
            'X_train': np.concatenate((data['X_train'][:num_train], data['X_val']), axis=0),
            'y_train': np.concatenate((data['y_train'][:num_train], data['y_val']), axis=0),
            'X_val': data['X_test'],
            'y_val': data['y_test'],
        }
    else:
        small_data = {
            'X_train': data['X_train'][:num_train],
            'y_train': data['y_train'][:num_train],
            'X_val': data['X_val'],
            'y_val': data['y_val'],
        }
    for reg_strenght in reg_strenghts:
        for update_rule in update_rules:
            print(f'running adaptive with {update_rule}')
            original_model, adaptive_model = get_models(args, reg_strenght)
            # todo: changed from Solver to AdaptiveSolver
            adaptive_solver = AdaptiveSolver(adaptive_model, small_data, print_every=args.print_every,
                                             num_epochs=args.epochs, batch_size=args.batch_size,
                                             update_rule=update_rule,
                                             optim_config={
                                                 'learning_rate': learning_rates[update_rule],
                                             },
                                             verbose=args.verbose,
                                             logger=Task.current_task().get_logger() if args.trains else None,
                                             eval_distribution_sample=args.eval_distribution_sample)
            # adaptive_solver = Solver(original_model, small_data, print_every=args.print_every,
            #                 num_epochs=args.epochs, batch_size=args.batch_size,
            #                 update_rule=update_rule,
            #                 optim_config={
            #                     'learning_rate': learning_rates[update_rule],
            #                 },
            #                 verbose=args.verbose)
            adaptive_solvers[update_rule] = adaptive_solver
            if not args.eval_distribution_sample:
                adaptive_solver.meta_train()
                # adaptive_solver.train()
            else:
                adaptive_solver.meta_train_eval_distribuiton_sample()
            print()
            print('Train result: train acc: %f; val_acc: %f' % (
                adaptive_solver.best_train_acc, adaptive_solver.best_val_acc))
            
            
            print(f'running with {update_rule}')
            solver = Solver(original_model, small_data, print_every=args.print_every,
                            num_epochs=args.epochs, batch_size=args.batch_size,
                            update_rule=update_rule,
                            optim_config={
                                'learning_rate': learning_rates[update_rule],
                            },
                            verbose=args.verbose)
            solvers[update_rule] = solver
            if args.baseline_as_well:
                solver.train()
                print('Train result: train acc: %f; val_acc: %f' % (
                    solver.best_train_acc, solver.best_val_acc))
                print()


            # plt.subplot(3, 1, 1)
            # plt.title('Training loss')
            # plt.xlabel('Iteration')
            #
            # plt.subplot(3, 1, 2)
            # plt.title('Training accuracy')
            # plt.xlabel('Epoch')
            #
            # plt.subplot(3, 1, 3)
            # plt.title('Validation accuracy')
            # plt.xlabel('Epoch')

            # for update_rule in ['sgd', 'sgd_momentum', 'adam', 'rmsprop']:

        plt.subplot(3, 1, 1)
        plt.title(f'Training loss. \n Reg: {reg_strenght}, Num trains: {num_train},')# LR: {lr}')
        plt.xlabel('Iteration')

        plt.subplot(3, 1, 2)
        plt.title('Training accuracy')
        plt.xlabel('Epoch')

        plt.subplot(3, 1, 3)
        plt.title('Validation accuracy')
        plt.xlabel('Epoch')

        if args.baseline_as_well:
            for update_rule, solver in solvers.items():
                plt.subplot(3, 1, 1)
                plt.plot(representation(solver.loss_history), 'o', label="loss_%s" % update_rule)

                # result_dict[(reg_strenghts, num_trains, update_rule, 'nonadaptive', 'loss_history')] = representation(solver.loss_history)

                plt.subplot(3, 1, 2)
                plt.plot(solver.train_acc_history, 'o', label="train_acc_%s" % update_rule)

                # result_dict[(reg_strenghts, num_trains, update_rule, 'nonadaptive', 'train_acc_history')] = solver.train_acc_history

                plt.subplot(3, 1, 3)

                plt.plot(solver.val_acc_history, 'o', label="val_acc_%s" % update_rule)

                # result_dict[(reg_strenght, num_train, update_rule, 'nonadaptive')] = solver

                best_val_acc, best_experiment = np.max(solver.val_acc_history), \
                                                np.argmax(solver.val_acc_history)
                result_dict[(num_train, reg_strenght, update_rule, 'nonadaptive',
                             best_val_acc, solver.train_acc_history[best_experiment], solver.loss_history[best_experiment])] = solver
        for update_rule, solver in adaptive_solvers.items():
            plt.subplot(3, 1, 1)
            plt.plot(representation(solver.loss_history), '-o', label="adaptive_loss_%s" % update_rule)

            plt.subplot(3, 1, 2)
            plt.plot(solver.train_acc_history, '-o', label="adaptive_train_acc_%s" % update_rule)

            # result_dict[(reg_strenghts, num_trains, update_rule, 'adaptive', 'train_acc_history')] = solver.train_acc_history

            plt.subplot(3, 1, 3)
            plt.plot(solver.val_acc_history, '-o', label="adaptive_val_acc_%s" % update_rule)
            best_val_acc, best_experiment = np.max(solver.val_acc_history), \
                                            np.argmax(solver.val_acc_history)
            result_dict[(num_train, reg_strenght, update_rule, 'adaptive',
                         best_val_acc, solver.train_acc_history[best_experiment], solver.loss_history[best_experiment])] = solver
        for i in [1, 2, 3]:
            plt.subplot(3, 1, i)
            plt.legend(loc='upper center', ncol=4)
        plt.gcf().set_size_inches(15, 15)
        plt.show()
    print(f"Best results, num train: {num_train}:")
    best_nonadaptive_descriotion = None
    best_nonadaptive_solver = None
    best_nonadaptive_val_acc = 0

    best_adaptive_descriotion = None
    best_adaptive_solver = None
    best_adaptive_val_acc = 0

    for desctiption, solver in result_dict.items():
        print(f"val acc history {solver.val_acc_history}")
        val_acc = np.max(solver.val_acc_history)  #[-1]
        print(f"{desctiption}, val_acc {val_acc}")
        if desctiption[3] == 'nonadaptive':
            if val_acc > best_nonadaptive_val_acc:
                best_nonadaptive_descriotion = desctiption
                best_nonadaptive_solver = solver
                best_nonadaptive_val_acc = val_acc
        else:
            if val_acc > best_adaptive_val_acc:
                # print(val_acc)
                best_adaptive_descriotion = desctiption
                best_adaptive_solver = solver
                best_adaptive_val_acc = val_acc
    if args.baseline_as_well:
        print("Best Nonadaptive solver:")
        best_val_acc, best_experiment = np.max(best_nonadaptive_solver.val_acc_history), \
                                        np.argmax(best_nonadaptive_solver.val_acc_history)
        print(f"Val acc: {best_nonadaptive_val_acc},"
              f" Train acc: {best_nonadaptive_solver.train_acc_history[best_experiment]},"
              f" loss: {best_nonadaptive_solver.loss_history[best_experiment]}")
        print(best_nonadaptive_descriotion)
    print("Best Adaptive solver:")
    best_val_acc, best_experiment = np.max(best_adaptive_solver.val_acc_history), \
                                    np.argmax(best_adaptive_solver.val_acc_history)
    print(f"Val acc: {best_adaptive_val_acc},"
          f" Train acc: {best_adaptive_solver.train_acc_history[best_experiment]},"
          f" loss: {best_adaptive_solver.loss_history[best_experiment]}")
    print(best_adaptive_descriotion)
    columns = ["Number of Trains", "Regularization",
               "Optimizer", "Adaptive?", "Validation acc", "Train acc", "Loss"]
    table = pd.DataFrame(result_dict.keys(), columns=columns)
    if args.trains:
        task.get_logger().report_table(title='Accuracy', series='Accuracy',
                           iteration=num_train, table_plot=table)
    # print(table)

    print(tabulate(table, headers=columns))
    return table


def mean_and_ci_result(args):
    if args.trains:
        task = Task.get_task(project_name='Flexible Regularization', task_name='Simple CNN')
        task = Task.init()
    tables = []
    for _ in range(args.num_of_repeats):
        tables.append(train_and_eval(args))
    pd.concat(tables)
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
    if args.trains:
        task.get_logger().report_table(title='Mean values', series='Mean values',
                       iteration=args.num_trains, table_plot=mean_values)
    print("standard deviation")
    print(tabulate(std, headers=std.columns))
    if args.trains:
        task.get_logger().report_table(title='Standard deviation', series='Standard deviation',
                                   iteration=args.num_trains, table_plot=std)
    
    
if __name__ == "__main__":
    args = parse_args()
    mean_and_ci_result(args)
    # train_and_eval(args)