import numpy as np
import theano
import theano.tensor as T
import time
import os

logit = lambda u: T.log(u / (1.-u))
logit_np = lambda u: np.log(u / (1.-u)).astype(theano.config.floatX)

from blocks.theano_expressions import l2_norm
def get_norms(model, gradients):
    """Compute norm of weights and their gradients divided by the number of elements"""
    norms = []
    grad_norms = []
    for param_name, param in model.params.iteritems():
        norm = T.sqrt(T.sum(T.square(param))) / T.prod(param.shape.astype(theano.config.floatX))
        norm.name = 'norm_' + param_name
        norms.append(norm)
        grad = gradients[param]
        #l2_norm(grad) - doesn't work due to unknown shape
        grad_norm = T.sqrt(T.sum(T.square(grad))) / T.prod(grad.shape.astype(theano.config.floatX))
        grad_norm.name = 'grad_norm_' + param_name
        grad_norms.append(grad_norm)
    return norms, grad_norms

def create_log_dir(args, model_id):
    model_id += args.suffix + time.strftime('-%y%m%dT%H%M%S')
    model_dir = os.path.join(os.path.expanduser(args.output_dir), model_id)
    os.makedirs(model_dir)
    return model_dir
