import argparse
import numpy as np
import os
import warnings

import theano
import theano.tensor as T

from blocks.algorithms import (RMSProp, GradientDescent, CompositeRule,
    RemoveNotFinite)
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.training import SharedVariableModifier
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import INPUT, PARAMETER

from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from fuel.transformers import Flatten, ScaleAndShift

import extensions
import model
import util

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=512, type=int,
                        help='Batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Initial learning rate. ' + \
                        'Will be decayed until it\'s 1e-5.')
    parser.add_argument('--resume_file', default=None, type=str,
                        help='Name of saved model to continue training')
    parser.add_argument('--suffix', default='', type=str,
                        help='Optional descriptive suffix for model')
    parser.add_argument('--output-dir', type=str, default='./',
                        help='Output directory to store trained models')
    parser.add_argument('--ext-every-n', type=int, default=25,
                        help='Evaluate training extensions every N epochs')
    parser.add_argument('--model-args', type=str, default='',
                        help='Dictionary string to be eval()d containing model arguments.')
    parser.add_argument('--dropout_rate', type=float, default=0.,
                        help='Rate to use for dropout during training+testing.')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='Name of dataset to use.')
    parser.add_argument('--plot_before_training', type=bool, default=False,
                        help='Save diagnostic plots at epoch 0, before any training.')
    args = parser.parse_args()

    model_args = eval('dict(' + args.model_args + ')')
    print model_args

    if not os.path.exists(args.output_dir):
        raise IOError("Output directory '%s' does not exist. "%args.output_dir)
    return args, model_args


if __name__ == '__main__':
    args, model_args = parse_args()

    if args.resume_file is not None:
        print "Resuming training from " + args.resume_file
        from blocks.scripts import continue_training
        continue_training(args.resume_file)

    ## load the training data
    if args.dataset == 'MNIST':
        from fuel.datasets import MNIST
        dataset_train = MNIST(['train'], sources=('features',))
        dataset_test = MNIST(['test'], sources=('features',))
        n_colors = 1
        spatial_width = 28
    elif args.dataset == 'CIFAR10':
        from fuel.datasets import CIFAR10
        dataset_train = CIFAR10(['train'], sources=('features',))
        dataset_test = CIFAR10(['test'], sources=('features',))
        n_colors = 3
        spatial_width = 32
    else:
        raise ValueError("Unknown dataset %s."%args.dataset)

    train_stream = Flatten(DataStream.default_stream(dataset_train,
                              iteration_scheme=ShuffledScheme(
                                  examples=dataset_train.num_examples,
                                  batch_size=args.batch_size)))
    test_stream = Flatten(DataStream.default_stream(dataset_test,
                             iteration_scheme=ShuffledScheme(
                                 examples=dataset_test.num_examples,
                                 batch_size=args.batch_size))
                             )

    # make the training data 0 mean and variance 1
    # TODO compute mean and variance on full dataset, not minibatch
    Xbatch = next(train_stream.get_epoch_iterator())[0]
    scl = 1./np.sqrt(np.mean((Xbatch-np.mean(Xbatch))**2))
    shft = -np.mean(Xbatch*scl)
    # scale is applied before shift
    train_stream = ScaleAndShift(train_stream, scl, shft)
    test_stream = ScaleAndShift(test_stream, scl, shft)

    ## initialize the model
    dpm = model.DiffusionModel(spatial_width, n_colors, **model_args)
    dpm.initialize()

    ## set up optimization
    features = T.matrix('features', dtype=theano.config.floatX)
    cost = dpm.cost(features)
    blocks_model = Model(cost)
    cg_nodropout = ComputationGraph(cost)
    if args.dropout_rate > 0:
        # DEBUG this triggers an error on my machine
        # apply dropout to all the input variables
        inputs = VariableFilter(roles=[INPUT])(cg_nodropout.variables)
        # dropconnect
        # inputs = VariableFilter(roles=[PARAMETER])(cg_nodropout.variables)
        cg = apply_dropout(cg_nodropout, inputs, args.dropout_rate)
    else:
        cg = cg_nodropout
    step_compute = RMSProp(learning_rate=args.lr, max_scaling=1e10)
    algorithm = GradientDescent(step_rule=CompositeRule([RemoveNotFinite(),
        step_compute]),
        params=cg.parameters, cost=cost)
    extension_list = []
    extension_list.append(
        SharedVariableModifier(step_compute.learning_rate,
            extensions.decay_learning_rate,
            after_batch=False,
            every_n_epochs=1, ))
    extension_list.append(FinishAfter(after_n_epochs=100001))

    ## set up logging
    extension_list.extend([Timing(), Printing()])
    model_dir = util.create_log_dir(args, dpm.name + '_' + args.dataset)
    model_save_name = os.path.join(model_dir, 'model.pkl')
    extension_list.append(
        Checkpoint(model_save_name, every_n_epochs=args.ext_every_n, save_separately=['log']))
    # generate plots
    extension_list.append(extensions.PlotMonitors(model_dir, every_n_epochs=args.ext_every_n))
    test_batch = next(test_stream.get_epoch_iterator())[0]
    extension_list.append(extensions.PlotSamples(dpm, algorithm, test_batch, model_dir,
        every_n_epochs=args.ext_every_n, before_training=args.plot_before_training))
    internal_state = dpm.internal_state(features)
    train_batch = next(train_stream.get_epoch_iterator())[0]
    extension_list.append(
        extensions.PlotInternalState(dpm, blocks_model, internal_state, features, train_batch, model_dir,
            every_n_epochs=args.ext_every_n, before_training=args.plot_before_training))
    extension_list.append(
        extensions.PlotParameters(dpm, blocks_model, model_dir,
            every_n_epochs=args.ext_every_n, before_training=args.plot_before_training))
    extension_list.append(
        extensions.PlotGradients(dpm, blocks_model, algorithm, train_batch, model_dir,
            every_n_epochs=args.ext_every_n, before_training=args.plot_before_training))
    # console monitors
    # # DEBUG -- incorporating train_monitor or test_monitor triggers a large number of
    # # float64 vs float32 GPU warnings, although monitoring still works. I think this is a Blocks
    # # bug. Uncomment this code to have more information during debugging/development.
    # train_monitor_vars = [cost]
    # norms, grad_norms = util.get_norms(blocks_model, algorithm.gradients)
    # train_monitor_vars.extend(norms + grad_norms)
    # train_monitor = TrainingDataMonitoring(
    #     train_monitor_vars, prefix='train', after_batch=True, before_training=True)
    # extension_list.append(train_monitor)
    # test_monitor_vars = [cost]
    # test_monitor = DataStreamMonitoring(test_monitor_vars, test_stream, prefix='test')
    # extension_list.append(test_monitor)

    ## train
    main_loop = MainLoop(model=blocks_model, algorithm=algorithm,
                         data_stream=train_stream,
                         extensions=extension_list)
    main_loop.run()

