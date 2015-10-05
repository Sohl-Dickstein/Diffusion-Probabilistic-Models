"""
Defines the function approximators
"""

import numpy as np
import theano.tensor as T

from blocks.bricks import Activation, MLP, Initializable, application, Identity
from blocks.bricks.conv import ConvolutionalActivation
from blocks.initialization import IsotropicGaussian, Constant, Orthogonal

# TODO IsotropicGaussian init will be wrong scale for some layers

class LeakyRelu(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return T.switch(input_ > 0, input_, 0.05*input_)

dense_nonlinearity = LeakyRelu()
conv_nonlinearity = LeakyRelu()

class MultiScaleConvolution(Initializable):
    def __init__(self, num_channels, num_filters, spatial_width, num_scales, filter_size, downsample_method='meanout', name=""):
        """
        A brick implementing a single layer in a multi-scale convolutional network.
        """
        super(MultiScaleConvolution, self).__init__()

        self.num_scales = num_scales
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.spatial_width = spatial_width
        self.downsample_method = downsample_method
        self.children = []

        print "adding MultiScaleConvolution layer"

        # for scale in range(self.num_scales-1, -1, -1):
        for scale in range(self.num_scales):
            print "scale %d"%scale
            conv_layer = ConvolutionalActivation(activation=conv_nonlinearity.apply,
                filter_size=(filter_size,filter_size), num_filters=num_filters,
                num_channels=num_channels, image_size=(spatial_width/2**scale, spatial_width/2**scale),
                # assume images are spatially smooth -- in which case output magnitude scales with
                # # filter pixels rather than square root of # filter pixels, so initialize
                # accordingly.
                weights_init=IsotropicGaussian(std=np.sqrt(1./(num_filters))/filter_size**2),
                biases_init=Constant(0), border_mode='full', name=name+"scale%d"%scale)
            self.children.append(conv_layer)

    def downsample(self, imgs_in, scale):
        """
        Downsample an image by a factor of 2**scale
        """
        imgs = imgs_in.copy()

        if scale == 0:
            return imgs

        # if self.downsample_method == 'maxout':
        #     print "maxout",
        #     imgs_maxout = downsample.max_pool_2d(imgs.copy(), (2**scale, 2**scale), ignore_border=False)
        # else:
        #     print "meanout",
        #     imgs_maxout = self.downsample_mean_pool_2d(imgs.copy(), (2**scale, 2**scale))

        num_imgs = imgs.shape[0].astype('int16')
        num_layers = imgs.shape[1].astype('int16')
        nlx0 = imgs.shape[2].astype('int16')
        nlx1 = imgs.shape[3].astype('int16')

        scalepow = np.int16(2**scale)

        # downsample
        imgs = imgs.reshape((num_imgs, num_layers, nlx0/scalepow, scalepow, nlx1/scalepow, scalepow))
        imgs = T.mean(imgs, axis=5)
        imgs = T.mean(imgs, axis=3)
        return imgs

    @application
    def apply(self, X):

        print "MultiScaleConvolution apply"

        nsamp = X.shape[0].astype('int16')

        Z = 0
        overshoot = (self.filter_size - 1)/2
        imgs_accum = 0 # accumulate the output image
        for scale in range(self.num_scales-1, -1, -1):
            # downsample image to appropriate scale
            imgs_down = self.downsample(X, scale)
            # do a convolutional transformation on it
            conv_layer = self.children[scale]
            # NOTE this is different than described in the paper, since each conv_layer
            # includes a nonlinearity -- it's not just one nonlinearity at the end
            imgs_down_conv = conv_layer.apply(imgs_down)

            # crop the edge so it's the same size as the input at that scale
            imgs_down_conv_croppoed = imgs_down_conv[:,:,overshoot:-overshoot,overshoot:-overshoot]
            imgs_accum += imgs_down_conv_croppoed

            if scale > 0:
                # scale up by factor of 2
                layer_width = self.spatial_width/2**scale
                imgs_accum = imgs_accum.reshape((nsamp, self.num_filters, layer_width, 1, layer_width, 1))
                imgs_accum = T.concatenate((imgs_accum, imgs_accum), axis=5)
                imgs_accum = T.concatenate((imgs_accum, imgs_accum), axis=3)
                imgs_accum = imgs_accum.reshape((nsamp, self.num_filters, layer_width*2, layer_width*2))

        return imgs_accum/self.num_scales


class MultiLayerConvolution(Initializable):
    def __init__(self, n_layers, n_hidden, spatial_width, n_colors, n_scales, filter_size=3):
        """
        A brick implementing a multi-layer, multi-scale convolutional network.
        """
        super(MultiLayerConvolution, self).__init__()

        self.children = []
        num_channels = n_colors
        for ii in xrange(n_layers):
            conv_layer = MultiScaleConvolution(num_channels, n_hidden, spatial_width, n_scales, filter_size, name="layer%d_"%ii)
            self.children.append(conv_layer)
            num_channels = n_hidden

    @application
    def apply(self, X):
        Z = X
        for conv_layer in self.children:
            Z = conv_layer.apply(Z)
        return Z

class MLP_conv_dense(Initializable):
    def __init__(self, n_layers_conv, n_layers_dense_lower, n_layers_dense_upper,
        n_hidden_conv, n_hidden_dense_lower, n_hidden_dense_lower_output, n_hidden_dense_upper,
        spatial_width, n_colors, n_scales, n_temporal_basis):
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
        self.mlp_conv = MultiLayerConvolution(n_layers_conv, n_hidden_conv, spatial_width, n_colors, n_scales)
        self.children = [self.mlp_conv]
        if n_hidden_dense_lower > 0 and n_layers_dense_lower > 0:
            n_input = n_colors*spatial_width**2
            n_output = n_hidden_dense_lower_output*spatial_width**2
            self.mlp_dense_lower = MLP([dense_nonlinearity] * n_layers_conv,
                [n_input] + [n_hidden_dense_lower] * (n_layers_conv-1) + [n_output],
                name='MLP dense lower', weights_init=Orthogonal(), biases_init=Constant(0))
            self.children.append(self.mlp_dense_lower)
        else:
            n_hidden_dense_lower_output = 0

        ## the upper layers (applied to each pixel independently)
        n_output = n_colors*n_temporal_basis*2 # "*2" for both mu and sigma
        self.mlp_dense_upper = MLP([dense_nonlinearity] * (n_layers_dense_upper-1) + [Identity()],
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
            n_images = X.shape[0].astype('int16')
            X = X.reshape((n_images, self.n_colors*self.spatial_width**2))
            Y_dense = self.mlp_dense_lower.apply(X)
            Y_dense = Y_dense.reshape((n_images, self.spatial_width, self.spatial_width,
                self.n_hidden_dense_lower_output))
            Y = T.concatenate([Y/T.sqrt(self.n_hidden_conv),
                Y_dense/T.sqrt(self.n_hidden_dense_lower_output)], axis=3)
        Z = self.mlp_dense_upper.apply(Y)
        return Z
