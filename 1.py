
import requests
import os
import mxnet as mx
import numpy as np
import math
import logging
from matplotlib import pyplot as plt
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

gpus = [0, 1, 2, 3, 4, 5, 6, 7]
context = [mx.gpu(i) for i in gpus]


# different levels of perturbation for the adversary (will be using FGSM to generate adversarial examples).
epsilons = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
# the fraction of samples to be drawn from each activation map for SAP
frac = 1.0
# the number of MC samples to estimate output of the model
mc_samples_output = 100
# the number of MC samples to estimate gradient of the model
mc_samples_gradient = 100


def download_file(url, local_fname):
    dir_name = os.path.dirname(local_fname)
    if dir_name != "":
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

    r = requests.get(url, stream=True)
    assert r.status_code == 200, "failed to open %s" % url
    with open(local_fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

train_fname = './data/cifar10_train.rec'
val_fname = './data/cifar10_val.rec'
download_file('http://data.mxnet.io/data/cifar10/cifar10_train.rec', train_fname)
download_file('http://data.mxnet.io/data/cifar10/cifar10_val.rec', val_fname)


class Uint8Iter(mx.io.DataIter):

    def __init__(self, iterator):
        self.iterator = iterator

    def next(self):
        batch = self.iterator.next()
        data = batch.data
        for i in range(len(data)):
            data[i] = data[i].astype('uint8').astype('float32')
        return mx.io.DataBatch(data = data, label = batch.label)

    def reset(self):
        self.iterator.reset()

    @property
    def provide_data(self):
        return self.iterator.provide_data

    @property
    def provide_label(self):
        return self.iterator.provide_label

def get_iter(batch_size):

    # training iterator
    train = mx.io.ImageRecordIter(
        path_imgrec         = train_fname,
        label_width         = 1,
        mean_r              = 123.68,
        mean_g              = 116.779,
        mean_b              = 103.939,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = (3, 28, 28),
        batch_size          = batch_size,
        rand_crop           = 1,
        max_random_scale    = 1,
        pad                 = 4,
        fill_value          = 127,
        min_random_scale    = 1,
        max_aspect_ratio    = 0,
        random_h            = 36,
        random_s            = 50,
        random_l            = 50,
        max_rotate_angle    = 0,
        max_shear_ratio     = 0,
        rand_mirror         = 1,
        shuffle             = True)

    # validation iterator
    val = mx.io.ImageRecordIter(
        path_imgrec         = val_fname,
        label_width         = 1,
        mean_r              = 123.68,
        mean_g              = 116.779,
        mean_b              = 103.939,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = (3, 28, 28),
        rand_crop           = False,
        rand_mirror         = False,
        shuffle             = True)

    return (Uint8Iter(train), Uint8Iter(val))

def sap_unit(data, frac, data_shape):
    shape = data.infer_shape(data=data_shape)[1][0]
    act = mx.sym.flatten(data)
    prob = mx.sym.broadcast_div(act, mx.sym.sum(act, axis=1).reshape((shape[0], 1)))
    prune = mx.sym.StochasticActivationPruning(act, prob, frac=frac)
    return prune.reshape(shape)

def residual_unit(data, num_filter, stride, dim_match, name, typ='normal', frac=1.0, shape=()):
    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=0.9, eps=2e-5, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_act1')
    if typ == 'sap':
        act1 = sap_unit(act1, frac, shape)
    conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1), no_bias=True, workspace=256, name=name + '_conv1')
    bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=0.9, eps=2e-5, name=name + '_bn2')
    act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name+'_act2')
    if typ == 'sap':
        act2 = sap_unit(act2, frac, shape)
    conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True, workspace=256, name=name + '_conv2')
    if dim_match:
        shortcut = data
    else:
        shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, no_bias=True, workspace=256, name=name + '_sc')
    return conv2 + shortcut

def get_symbol(typ='dense', frac=1.0, shape=()):

    filter_list = [16, 16, 32, 64]

    data = mx.sym.Variable('data')
    data = mx.sym.identity(data=data, name='id')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=0.9, name='bn_data')
    body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1), no_bias=True, workspace=256, name='conv0')

    for i in range(3):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False, 'stage%d_unit%d' % (i + 1, 1), typ, frac, shape)
        for j in range(2):
            body = residual_unit(body, filter_list[i+1], (1, 1), True, 'stage%d_unit%d' % (i + 1, j + 2), typ, frac, shape)

    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=0.9, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    if typ == 'sap':
        relu1 = sap_unit(relu1, frac, shape)
    pool1 = mx.sym.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.sym.Flatten(data=pool1)
    fc1 = mx.sym.FullyConnected(data=flat, num_hidden=10, name='fc1')
    
    return mx.sym.SoftmaxOutput(data=fc1, name='softmax')


# fixed paramters
label_name = 'softmax_label'
eval_metric = 'acc'
lr = 0.5
num_epoch = 150
batch_size = 512
lr_steps = [100, 130]
lr_factor = 0.1
batches_per_epoch = math.ceil(50000.0 / batch_size)
mom = 0.9
wd = 0.0001
initializer = mx.init.Xavier(rnd_type='gaussian', factor_type='in', magnitude=2)
disp_batches = 500

# get data iterators
train, val = get_iter(batch_size)

# train model
sym = get_symbol()
lr_steps = [batches_per_epoch * i for i in lr_steps]
lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=lr_steps, factor=lr_factor)
optimizer_params = {
    'learning_rate' : lr,
    'lr_scheduler'  : lr_scheduler,
    'momentum'      : mom,
    'wd'            : wd}
batch_end_callbacks = [mx.callback.Speedometer(batch_size, disp_batches)]
mod = mx.mod.Module(symbol=sym, context=context, label_names=[label_name,])
mod.fit(train,
    num_epoch               = num_epoch,
    eval_data               = val,
    eval_metric             = eval_metric,
    optimizer_params        = optimizer_params,
    initializer             = initializer,
    batch_end_callback      = batch_end_callbacks,
    allow_missing           = True)

# save model
mod.save_params('./trained_model')


class AdversaryIter(mx.io.DataIter):

    def __init__(self, iterator, batch_size, mod, epsilons, frac):
        self.iterator = iterator
        self.batch_size = batch_size
        self.mod = mod
        self.epsilons = epsilons
        self.frac = frac

    def next(self):
        batch = self.iterator.next()
        data = batch.data
        self.mod.forward(batch)
        self.mod.backward()
        grad = self.mod.get_input_grads()
        for i in range(mc_samples_gradient - 1):
            self.mod.forward(batch)
            self.mod.backward()
            new_grad = self.mod.get_input_grads()
            for j in range(len(data)):
                grad[j] += new_grad[j]
        for i in range(len(data)):
            grad[i] /= float(mc_samples_gradient)
        new_data = [[matrix.copy() for matrix in data] for i in range(len(self.epsilons))]
        for i in range(len(self.epsilons)):
            for j in range(len(data)):
                noise = self.epsilons[i] * mx.nd.sign(grad[j].as_in_context(new_data[i][j].context))
                new_data[i][j] += noise
                new_data[i][j] = mx.nd.clip(new_data[i][j], 0, 255)
        return [mx.io.DataBatch(data = new_data[i], label = batch.label) for i in range(len(self.epsilons))]

    def reset(self):
        self.iterator.reset()

    @property
    def provide_data(self):
        return self.iterator.provide_data

    @property
    def provide_label(self):
        return self.iterator.provide_label

# fixed parameters
label_name = 'softmax_label'
eval_metric = 'acc'
batch_size = 500 * len(gpus)
shape = (500, 3, 28, 28)

# get data iterators
train, val = get_iter(batch_size)

def accuracy(iterator, mod, number):

    correct = [0.0 for i in range(number)]
    total = [0.0 for i in range(number)]

    iterator.reset()
    while True:
        try:
            batches = iterator.next()
        except StopIteration:
            break
        for j in range(number):
            batch = batches[j]

            data = batch.data
            mod.forward(batch)
            out = mod.get_outputs()[0].asnumpy()
            for i in range(mc_samples_output - 1):
                mod.forward(batch)
                out += mod.get_outputs()[0].asnumpy()
            out = out / float(mc_samples_output)

            label = batch.label[0].asnumpy()

            for i in range(out.shape[0]):
                index = np.argmax(out[i])
                total[j] += 1.0
                if index == label[i]:
                    correct[j] += 1.0

    acc = [0 for i in range(number)]
    for i in range(number):
        acc[i] = correct[i] / total[i]

    return acc

def validate(typ):

    # get model
    sym = get_symbol(typ, frac, shape)
    mod = mx.mod.Module(symbol=sym, context=context, label_names=[label_name,])
    mod.bind(val.provide_data, label_shapes = val.provide_label, for_training = False)
    mod.load_params('trained_model')

    # get model for adversarial examples
    adv_sym = get_symbol(typ, frac, shape)
    mod_adv = mx.mod.Module(symbol=adv_sym, context=context, label_names=[label_name,])
    mod_adv.bind(val.provide_data, label_shapes = val.provide_label, for_training = True, inputs_need_grad = True)
    mod_adv.load_params('trained_model')

    # validate model
    val.reset()
    new_val = AdversaryIter(val, batch_size, mod_adv, epsilons, frac)
    acc = accuracy(new_val, mod, len(epsilons))

    return acc

dense_acc = validate('dense')
sap_acc = validate('sap')

ax = plt.subplot(111)
x = range(8)
ax.plot(range(8), dense_acc, 'o', linestyle='-', c='k', label='DENSE')
ax.plot(range(8), sap_acc, 'o', linestyle='-', c='r', label='SAP-$100$')
plt.xlabel('$\lambda$')
plt.xticks(range(8), [0, 1, 2, 4, 8, 16, 32, 64])
plt.ylabel('Accuracy')
plt.ylim(0, 1)
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), fancybox=True, shadow=True, ncol=2)
plt.show()
