import numpy as np
import theano
import theano.tensor as T
import util

from blocks.bricks import (Activation, MLP, Initializable, Rectifier, Tanh, Random, application,
    Identity)
from blocks.bricks.conv import ConvolutionalActivation
from blocks.initialization import IsotropicGaussian, Constant, Orthogonal

# TODO IsotropicGaussian init will be wrong scale for some layers

class LeakyRelu(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return T.switch(input_ > 0, input_, 0.05*input_)

class MultiLayerConvolution(Initializable):
    def __init__(self, n_layers, n_hidden, spatial_width, n_colors, filter_size=3):
        """
        A brick implementing a multi-layer convolutional network.
        TODO make this multi-scale multi-layer convolution
        """
        super(MultiLayerConvolution, self).__init__()

        self.filter_size = filter_size
        self.children = []
        num_channels = n_colors
        for ii in xrange(n_layers):
            conv_layer = ConvolutionalActivation(activation=LeakyRelu().apply,
                filter_size=(filter_size,filter_size), num_filters=n_hidden,
                num_channels=num_channels, image_size=(spatial_width, spatial_width),
                # assume images are spatially smooth -- in which case output magnitude scales with
                # # filter pixels rather than square root of # filter pixels, so initialize 
                # accordingly.
                weights_init=IsotropicGaussian(std=np.sqrt(1./(n_hidden))/filter_size**2),
                biases_init=Constant(0), border_mode='full', name="conv%d"%ii)
            self.children.append(conv_layer)
            num_channels = n_hidden

    @application
    def apply(self, X):
        """
        Take in noisy input image and output temporal coefficients for mu and sigma.
        """
        Z = X
        overshoot = (self.filter_size - 1)/2
        for conv_layer in self.children:
            Z = conv_layer.apply(Z)
            Z = Z[:,:,overshoot:-overshoot,overshoot:-overshoot]
        return Z

class MLP_conv_dense(Initializable):
    def __init__(self, n_layers_conv, n_layers_dense_lower, n_layers_dense_upper,
        n_hidden_conv, n_hidden_dense_lower, n_hidden_dense_lower_output, n_hidden_dense_upper,
        spatial_width, n_colors, n_temporal_basis):
        """
        The multilayer perceptron, that provides temporal weighting coefficients for mu and sigma
        images. This consists of a lower segment with a convolutional MLP, and optionally with a 
        dense MLP in parallel. The upper segment then consists of a per-pixel dense MLP 
        (convolutional MLP with 1x1 kernel).
        """
        super(MLP_conv_dense, self).__init__()

        self.n_colors = n_colors
        self.spatial_width = spatial_width
        self.n_hidden_dense_lower = n_hidden_dense_lower
        self.n_hidden_dense_lower_output = n_hidden_dense_lower_output
        self.n_hidden_conv = n_hidden_conv

        ## the lower layers
        self.mlp_conv = MultiLayerConvolution(n_layers_conv, n_hidden_conv, spatial_width, n_colors)
        self.children = [self.mlp_conv]
        if n_hidden_dense_lower > 0 and n_layers_dense_lower > 0:
            n_input = n_colors*spatial_width**2
            n_output = n_hidden_dense_lower_output*spatial_width**2
            self.mlp_dense_lower = MLP([Tanh()] * n_layers_conv,
                [n_input] + [n_hidden_dense_lower] * (n_layers_conv-1) + [n_output],
                name='MLP dense lower', weights_init=Orthogonal(), biases_init=Constant(0))
            self.children.append(self.mlp_dense_lower)
        else:
            n_hidden_dense_lower_output = 0

        ## the upper layers (applied to each pixel independently)
        n_output = n_colors*n_temporal_basis*2 # "*2" for both mu and sigma
        self.mlp_dense_upper = MLP([Tanh()] * (n_layers_dense_upper-1) + [Identity()],
            [n_hidden_conv+n_hidden_dense_lower_output] + 
            [n_hidden_dense_upper] * (n_layers_dense_upper-1) + [n_output],
            name='MLP dense upper', weights_init=Orthogonal(), biases_init=Constant(0))
        self.children.append(self.mlp_dense_upper)

    @application
    def apply(self, X):
        """
        Take in noisy input image and output temporal coefficients for mu and sigma.
        """
        Y = self.mlp_conv.apply(X)
        Y = Y.dimshuffle(0,2,3,1)
        if self.n_hidden_dense_lower > 0:
            n_images = X.shape[0]
            X = X.reshape((n_images, self.n_colors*self.spatial_width**2))
            Y_dense = self.mlp_dense_lower.apply(X)
            Y_dense = Y_dense.reshape((n_images, self.spatial_width, self.spatial_width,
                self.n_hidden_dense_lower_output))
            Y = T.concatenate([Y/T.sqrt(self.n_hidden_conv),
                Y_dense/T.sqrt(self.n_hidden_dense_lower_output)], axis=3)
        Z = self.mlp_dense_upper.apply(Y)
        return Z
