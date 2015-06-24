"""
Extensions called during training to generate samples and diagnostic plots and printouts.
"""

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import theano.tensor as T
import theano

from blocks.extensions import SimpleExtension

import viz
import sampler


class PlotSamples(SimpleExtension):
    def __init__(self, model, algorithm, X, path, n_samples=49, **kwargs):
        """
        Generate samples from the model. The do() function is called as an extension during training.
        Generates 3 types of samples:
        - Sample from generative model
        - Sample from image denoising posterior distribution (default signal to noise of 1)
        - Sample from image inpainting posterior distribution (inpaint left half of image)
        """

        super(PlotSamples, self).__init__(**kwargs)
        self.model = model
        self.path = path
        self.X = X[:n_samples].reshape(
            (n_samples, model.n_colors, model.spatial_width, model.spatial_width))
        self.n_samples = n_samples
        X_noisy = T.tensor4('X noisy samp')
        t = T.matrix('t samp')
        self.get_mu_sigma = theano.function([X_noisy, t], model.get_mu_sigma(X_noisy, t),
            allow_input_downcast=True)

    def do(self, callback_name, *args):
        print "generating samples"
        base_fname_part1 = self.path + '/samples-'
        base_fname_part2 = '_epoch%04d'%self.main_loop.status['epochs_done']
        sampler.generate_samples(self.model, self.get_mu_sigma,
            n_samples=self.n_samples, inpaint=False, denoise_sigma=None, X_true=None,
            base_fname_part1=base_fname_part1, base_fname_part2=base_fname_part2)
        sampler.generate_samples(self.model, self.get_mu_sigma,
            n_samples=self.n_samples, inpaint=True, denoise_sigma=None, X_true=self.X,
            base_fname_part1=base_fname_part1, base_fname_part2=base_fname_part2)
        sampler.generate_samples(self.model, self.get_mu_sigma,
            n_samples=self.n_samples, inpaint=False, denoise_sigma=1, X_true=self.X,
            base_fname_part1=base_fname_part1, base_fname_part2=base_fname_part2)


class PlotParameters(SimpleExtension):
    def __init__(self, model, blocks_model, path, **kwargs):
        super(PlotParameters, self).__init__(**kwargs)
        self.path = path
        self.model = model
        self.blocks_model = blocks_model

    def do(self, callback_name, *args):
        print "plotting parameters"
        for param_name, param in self.blocks_model.params.iteritems():
            filename_safe_name = '-'.join(param_name.split('/')[2:]).replace(' ', '_')
            base_fname_part1 = self.path + '/params-' + filename_safe_name
            base_fname_part2 = '_epoch%04d'%self.main_loop.status['epochs_done']
            viz.plot_parameter(param.get_value(), base_fname_part1, base_fname_part2,
                title=param_name, n_colors=self.model.n_colors)


class PlotGradients(SimpleExtension):
    def __init__(self, model, blocks_model, algorithm, X, path, **kwargs):
        super(PlotGradients, self).__init__(**kwargs)
        self.path = path
        self.X = X
        self.model = model
        self.blocks_model = blocks_model
        gradients = []
        for param_name in sorted(self.blocks_model.params.keys()):
            gradients.append(algorithm.gradients[self.blocks_model.params[param_name]])
        self.grad_f = theano.function(algorithm.inputs, gradients, allow_input_downcast=True)

    def do(self, callback_name, *args):
        print "plotting gradients"
        grad_vals = self.grad_f(self.X)
        keynames = sorted(self.blocks_model.params.keys())
        for ii in xrange(len(keynames)):
            param_name = keynames[ii]
            val = grad_vals[ii]
            filename_safe_name = '-'.join(param_name.split('/')[2:]).replace(' ', '_')
            base_fname_part1 = self.path + '/grads-' + filename_safe_name
            base_fname_part2 = '_epoch%04d'%self.main_loop.status['epochs_done']
            viz.plot_parameter(val, base_fname_part1, base_fname_part2,
                title="grad " + param_name, n_colors=self.model.n_colors)


class PlotInternalState(SimpleExtension):
    def __init__(self, model, blocks_model, state, features, X, path, **kwargs):
        super(PlotInternalState, self).__init__(**kwargs)
        self.path = path
        self.X = X
        self.model = model
        self.blocks_model = blocks_model
        self.internal_state_f = theano.function([features], state, allow_input_downcast=True)
        self.internal_state_names = []
        for var in state:
            self.internal_state_names.append(var.name)

    def do(self, callback_name, *args):
        print "plotting internal state of network"
        state = self.internal_state_f(self.X)
        for ii in xrange(len(state)):
            param_name = self.internal_state_names[ii]
            val = state[ii]
            filename_safe_name = param_name.replace(' ', '_').replace('/', '-')
            base_fname_part1 = self.path + '/state-' + filename_safe_name
            base_fname_part2 = '_epoch%04d'%self.main_loop.status['epochs_done']
            viz.plot_parameter(val, base_fname_part1, base_fname_part2,
                title="state " + param_name, n_colors=self.model.n_colors)


class PlotMonitors(SimpleExtension):
    def __init__(self, path, burn_in_iters=0, **kwargs):
        super(PlotMonitors, self).__init__(**kwargs)
        self.path = path
        self.burn_in_iters = burn_in_iters

    def do(self, callback_name, *args):
        print "plotting monitors"
        df = self.main_loop.log.to_dataframe()
        iter_number  = df.tail(1).index
        # Throw out the first burn_in values
        # as the objective is often much larger
        # in that period.
        if iter_number > self.burn_in_iters:
            df = df.loc[self.burn_in_iters:]
        cols = [col for col in df.columns if col.startswith(('cost', 'train', 'test'))]
        df = df[cols].interpolate(method='linear')

        # If we don't have any non-nan dataframes, don't plot
        if len(df) == 0:
            return
        try:
            axs = df.interpolate(method='linear').plot(
                subplots=True, legend=False, figsize=(5, len(cols)*2))
        except TypeError:
            # This starting breaking after updating Blocks.
            print "Failed to generate monitoring plots."
            return

        for ax, cname in zip(axs, cols):
            ax.set_title(cname)
        fn = os.path.join(self.path,
            'monitors_subplots_epoch%04d.png' % self.main_loop.status['epochs_done'])
        plt.savefig(fn, bbox_inches='tight')

        plt.clf()
        df.plot(subplots=False, figsize=(15,10))
        plt.gcf().tight_layout()
        fn = os.path.join(self.path,
            'monitors_epoch%04d.png' % self.main_loop.status['epochs_done'])
        plt.savefig(fn, bbox_inches='tight')
        plt.close('all')


def decay_learning_rate(iteration, old_value):
    # TODO the numbers in this function should not be hard coded

    # this is called every epoch
    # reduce the learning rate by 10 every 1000 epochs

    decay_rate = np.exp(np.log(0.1)/1000.)
    new_value = decay_rate*old_value
    if new_value < 1e-5:
        new_value = 1e-5
    print "learning rate %g"%new_value
    return np.float32(new_value)
