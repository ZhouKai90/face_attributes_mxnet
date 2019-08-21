import sys
import os
import mxnet as mx
from fa_config import fa_config as fc

blocks = [1, 4, 8, 2]

def Act(data, act_type, name):
    # ignore param act_type, set it in this function
    if act_type == 'prelu':
        body = mx.sym.LeakyReLU(data=data, act_type='prelu', name=name)
    else:
        body = mx.sym.Activation(data=data, act_type=fc.act_type, name=name)
    return body


def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride,
                              pad=pad, no_bias=True, name='%s%s_conv2d' % (name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' % (name, suffix), fix_gamma=False, momentum=fc.bn_mom)
    act = Act(data=bn, act_type=fc.act_type, name='%s%s_relu' % (name, suffix))
    return act


def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride,
                              pad=pad, no_bias=True, name='%s%s_conv2d' % (name, suffix))
    bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' % (name, suffix), fix_gamma=False, momentum=fc.bn_mom)
    return bn


def ConvOnly(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride,
                              pad=pad, no_bias=True, name='%s%s_conv2d' % (name, suffix))
    return conv


def DResidual(data, num_out=1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=1, name=None, suffix=''):
    conv = Conv(data=data, num_filter=num_group, kernel=(1, 1), pad=(0, 0), stride=(1, 1),
                name='%s%s_conv_sep' % (name, suffix))
    conv_dw = Conv(data=conv, num_filter=num_group, num_group=num_group, kernel=kernel, pad=pad, stride=stride,
                   name='%s%s_conv_dw' % (name, suffix))
    proj = Linear(data=conv_dw, num_filter=num_out, kernel=(1, 1), pad=(0, 0), stride=(1, 1),
                  name='%s%s_conv_proj' % (name, suffix))
    return proj


def Residual(data, num_block=1, num_out=1, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, name=None, suffix=''):
    identity = data
    for i in range(num_block):
        shortcut = identity
        conv = DResidual(data=identity, num_out=num_out, kernel=kernel, stride=stride, pad=pad, num_group=num_group,
                         name='%s%s_block' % (name, suffix), suffix='%d' % i)
        identity = conv + shortcut
    return identity


def get_symbol(args):
    data = mx.symbol.Variable(name="data")
    data = data - 127.5
    data = data * 0.0078125

    conv_1 = Conv(data, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name="conv_1")
    if blocks[0] == 1:
        conv_2_dw = Conv(conv_1, num_group=64, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                         name="conv_2_dw")
    else:
        conv_2_dw = Residual(conv_1, num_block=blocks[0], num_out=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                             num_group=64, name="res_2")
    conv_23 = DResidual(conv_2_dw, num_out=64, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=128, name="dconv_23")
    conv_3 = Residual(conv_23, num_block=blocks[1], num_out=64, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=128,
                      name="res_3")
    conv_34 = DResidual(conv_3, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=256, name="dconv_34")
    conv_4 = Residual(conv_34, num_block=blocks[2], num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                      num_group=256, name="res_4")
    conv_45 = DResidual(conv_4, num_out=128, kernel=(3, 3), stride=(2, 2), pad=(1, 1), num_group=512, name="dconv_45")
    conv_5 = Residual(conv_45, num_block=blocks[3], num_out=128, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                      num_group=256, name="res_5")
    conv_6_sep = Conv(conv_5, num_filter=512, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_6sep")
    conv_6_dw = Linear(conv_6_sep, num_filter=512, num_group=512, kernel=(7, 7), pad=(0, 0), stride=(1, 1), name="conv_6dw7_7")
    conv_6_f = mx.sym.FullyConnected(data=conv_6_dw, num_hidden=13, name='pre_fc1')
    fc1 = mx.sym.BatchNorm(data=conv_6_f, fix_gamma=True, eps=2e-5, momentum=fc.bn_mom, name='fc1')

    label = mx.symbol.Variable('softmax_label')
    gender_label = mx.symbol.slice_axis(data=label, axis=1, begin=0, end=1)
    gender_label = mx.symbol.reshape(gender_label, shape=(args.batch_size,))
    gender_fc1 = mx.symbol.slice_axis(data=fc1, axis=1, begin=0, end=2)

    mask_label = mx.symbol.slice_axis(data=label, axis=1, begin=1, end=2)
    mask_label = mx.symbol.reshape(mask_label, shape=(args.batch_size,))
    mask_fc1 = mx.symbol.slice_axis(data=fc1, axis=1, begin=2, end=4)

    glass_label = mx.symbol.slice_axis(data=label, axis=1, begin=2, end=3)
    glass_label = mx.symbol.reshape(glass_label, shape=(args.batch_size,))
    glass_fc1 = mx.symbol.slice_axis(data=fc1, axis=1, begin=4, end=7)

    mouth_label = mx.symbol.slice_axis(data=label, axis=1, begin=3, end=4)
    mouth_label = mx.symbol.reshape(mouth_label, shape=(args.batch_size,))
    mouth_fc1 = mx.symbol.slice_axis(data=fc1, axis=1, begin=7, end=10)

    eye_label = mx.symbol.slice_axis(data=label, axis=1, begin=4, end=5)
    eye_label = mx.symbol.reshape(eye_label, shape=(args.batch_size,))
    eye_fc1 = mx.symbol.slice_axis(data=fc1, axis=1, begin=10, end=13)

    gender_softmax = mx.symbol.SoftmaxOutput(data=gender_fc1, label=gender_label, name='gender_softmax',
                                             normalization='valid', use_ignore=True, ignore_label=9999)
    mask_softmax = mx.symbol.SoftmaxOutput(data=mask_fc1, label=mask_label, name='mask_softmax',
                                           normalization='valid', use_ignore=True, ignore_label=9999)
    glass_softmax = mx.symbol.SoftmaxOutput(data=glass_fc1, label=glass_label, name='glass_softmax',
                                            normalization='valid', use_ignore=True, ignore_label=9999)
    mouth_softmax = mx.symbol.SoftmaxOutput(data=mouth_fc1, label=mouth_label, name='mouth_softmax',
                                            normalization='valid', use_ignore=True, ignore_label=9999)
    eye_softmax = mx.symbol.SoftmaxOutput(data=eye_fc1, label=eye_label, name='eye_softmax', normalization='valid',
                                          use_ignore=True, ignore_label=9999)
    outs = [gender_softmax, mask_softmax, glass_softmax, mouth_softmax, eye_softmax]
    outs.append(mx.sym.BlockGrad(fc1))
    return mx.symbol.Group(outs)

if __name__ == '__main__':
    sym = get_symbol()
    allLayers = sym.get_internals()
    print(allLayers)
    mx.viz.plot_network(symbol = sym)