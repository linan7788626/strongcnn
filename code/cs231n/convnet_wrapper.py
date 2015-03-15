"""
from
https://piazza.com/class/i37qi08h43qfv?cid=452

ex:

# define the layers, layers are defined from input to output
layer_defs = [
    {
        'name': 'C0',
        'type': 'conv-relu-pool',
        'size': 3,
        'channels': int(sys.argv[1]),
    },

    {
        'name': 'C1',
        'type': 'conv-relu-pool',
        'size': 3,
        'channels': int(sys.argv[2]),
    },

    {
        'name': 'C2',
        'type': 'conv-relu',
        'size': 3,
        'channels': int(sys.argv[3]),
    },

    {
        'name': 'A0',
        'type': 'affine-relu',
        'outsize': int(sys.argv[4]),
    },

    {
        'name': 'A1',
        'type': 'affine',
        'outsize': num_classes,
    },
]

# initialize the network model and layers
model, layers = init_modular_convnet(layer_defs)
loss_func = modular_loss_function(layers)

# train my awesome neural network here using loss_func for training
# ...
# ...

# save the network to disk
save_modular_convnet(best_model, layers, 'layer_model.pickle')
"""

import cPickle as pickle

def raw_forward_function(layer_type):
    """
    Given a layer type as a string, return the corresponding forward function
    """
    forward_functions_dict = {
        'conv': conv_forward_fast,
        'relu': relu_forward,
        'pool': max_pool_forward_fast,
        'affine': affine_forward,
        'affine-relu': affine_relu_forward,
        'conv-relu': conv_relu_forward,
        'conv-relu-pool': conv_relu_pool_forward,
    }

    return forward_functions_dict[layer_type]

def raw_backward_function(layer_type):
    """
    Given a layer type as a string, return the corresponding backward function
    """
    backward_functions_dict = {
        'conv': conv_backward_fast,
        'relu': relu_backward,
        'pool': max_pool_backward_fast,
        'affine': affine_backward,
        'affine-relu': affine_relu_backward,
        'conv-relu': conv_relu_backward,
        'conv-relu-pool': conv_relu_pool_backward,
    }

    return backward_functions_dict[layer_type]

def extra_params(layer_type, layer_name, model):
    """
    Given a layer type as a string, return a tuple of extra parameters that
    should be passed to the forward or backward functions for those layers.
    Assumes that each conv layer input is padded by (HH-1)/2, and each pool
    layer is 2x2 in size
    """
    W = model['W' + layer_name]
    params = []
    if 'conv' in layer_type:
        HH = W.shape[2]
        assert HH % 2 == 1, 'filter must be of odd size'
        conv_param = {'stride': 1, 'pad': (HH - 1)/2}
        params.append(conv_param)

    if 'pool' in layer_type:
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        params.append(pool_param)

    return tuple(params)

def forward_function(layer_type, layer_name, model):
    """
    Returns a lambda function that can take as an input X and give the next
    layer output. This function takes the layer type, layer name and model
    dictionary as inputs
    """
    params = extra_params(layer_type, layer_name, model)
    func = raw_forward_function(layer_type)
    W = model['W' + layer_name]
    b = model['b' + layer_name]
    return lambda X: func(X, W, b, *params)

def backward_function(layer_type):
    """
    Same as before, but returns the corresponding backward function instead.
    Since the backward function does not require any parameters, this is a thin
    wrapper around raw_backward_function
    """
    func = raw_backward_function(layer_type)
    return lambda dX, cache: func(dX, cache)

class ModularLayer:
    """
    A modular layer interface class.
    Can be initialized from a layer_def that contains relevant parameters such
    as the convolution size for conv layers.
    Can also perform forward/backward passes over input data/backward ddata
    """
    def __init__(self, layer_def):
        self.layer_def = layer_def

    def forward(self, data, model):
        """
        Perform forward pass over a data given a model dict, and save the cache
        in self.cache
        """
        ff = forward_function(self.layer_def['type'],
                              self.layer_def['name'],
                              model)
        output, cache = ff(data)
        self.cache = cache
        return output

    def backward(self, ddata):
        """
        Perform backward pass over a given ddata. Uses the saved self.cache
        variable
        """
        bf = backward_function(self.layer_def['type'])
        din, dW, db = bf(ddata, self.cache)
        return din, dW, db

    def layer_name(self):
        return self.layer_def['name']

    def layer_type(self):
        return self.layer_def['type']

    def Wname(self):
        """
        The name of the weights for this layer in the model dict
        """
        return 'W' + self.layer_name()

    def bname(self):
        """
        The name of the biases for this layer in the model dict
        """
        return 'b' + self.layer_name()

    def Wdim(self, indim):
        """
        Returns the weight dimensions given the input data dimensions
        """
        if 'conv' in self.layer_type():
            HH = self.layer_def['size']
            channels = self.layer_def['channels']
            inchannels = indim[0]
            return (channels, inchannels, HH, HH)
        if 'affine' in self.layer_type():
            insize = np.prod(indim)
            outsize = self.layer_def['outsize']
            return (insize, outsize)
        assert 0, 'layer must be derived from affine or conv type'

    def bdim(self, indim):
        """
        Returns the bias dimensions given the input data dimensions
        """
        if 'conv' in self.layer_type():
            return (self.layer_def['channels'],)
        if 'affine' in self.layer_type():
            return (self.layer_def['outsize'],)

    def outdim(self, indim):
        """
        Returns the output dimensions given the input data dimensions
        """
        inH = indim[1]
        inW = indim[2]
        if 'conv' in self.layer_type():
            HH = self.layer_def['size']
            channels = self.layer_def['channels']
            outH = inH
            outW = inW
        elif 'affine' in self.layer_type():
            channels = 1
            outH = 1
            outW = self.layer_def['outsize']

        if 'pool' in self.layer_type():
            outH /= 2
            outW /= 2

        return (channels, outH, outW)


def modular_convnet(layers, X, model, y=None, reg=0.0, verbose=False):
    """
    A modular convnet generated from a model_def list
    layers is a list of ModularLayer objects
    The layers should be listed so that the layer closest to the input is at
    index 0 and arranged in the order of their depth
    """

    data = X
    for layer in layers:
        data = layer.forward(data, model)
    scores = data

    if y is None:
        return scores

    data_loss, ddata = softmax_loss(scores, y)

    grads = {}
    reg_loss = 0

    for layer in reversed(layers):
        ddata, dW, db = layer.backward(ddata)
        W = model[layer.Wname()]
        dW += reg * W
        reg_loss += 0.5 * reg * np.sum(W * W)
        grads[layer.Wname()] = dW
        grads[layer.bname()] = db

    loss = data_loss + reg_loss

    return loss, grads

def init_modular_convnet(layer_defs, weight_scale=5e-2, bias_scale=0,
                         input_shape=(3, 32, 32)):
    """
    Initialize a modular convnet given the layer definitions in layer_defs and
    other parameters as in the other init_*_convnet functions
    """
    model = {}

    # make sure that the layer names are all unique
    layer_names = [layer_def['name'] for layer_def in layer_defs]
    unique_layer_names = set(layer_names)
    assert len(layer_names) == len(unique_layer_names), """layer names should be
    unique"""

    layers = [ModularLayer(layer_def) for layer_def in layer_defs]

    indim = input_shape
    for layer in layers:
        Wdim = layer.Wdim(indim)
        bdim = layer.bdim(indim)
        outdim = layer.outdim(indim)
        model[layer.Wname()] = weight_scale * np.random.randn(*Wdim)
        model[layer.bname()] = bias_scale * np.random.randn(*bdim)
        indim = outdim

    return model, layers

def modular_loss_function(layers):
    """
    Return a lambda suitable for passing to train function
    """
    return lambda *args, **kw: modular_convnet(layers, *args, **kw)

def save_modular_convnet(model, layers, filename):
    """
    Save the modular convnet layer and model definitions into a file
    """
    with open(filename, 'w') as fp:
        pickle.dump((model, layers), fp)

def load_modular_convnet(filename):
    """
    Load the modular convnet layer and model definitions from a file
    """
    with open(filename) as fp:
        model, layers = pickle.load(fp)
        return model, layers
