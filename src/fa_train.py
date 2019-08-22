import argparse
import mxnet as mx
import logging
# import os, sys
# import numpy as np
from fa_dataset import get_dataIter_from_rec
from fa_config import fa_config as fc
from fa_eighth_mobile_net import get_symbol
from fa_eval_metric import ClassAccMetric


def get_fine_tune_model(sym, arg_params, num_classes, layer_name):
    
    all_layers = sym.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc')
    net = mx.symbol.sigmoid(data=net, name='sig')
    net = mx.symbol.Custom(data=net, name='softmax', op_type='CrossEntropyLoss')

    new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k})
    return (net, new_args)

def multi_factor_scheduler(begin_epoch, epoch_size, lr_steps, factor=0.1):
    steps = [int(x) for x in lr_steps.split(',')]
    step_ = [epoch_size * (x-begin_epoch) for x in steps if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None


def train_model(args, net):
    kv = mx.kvstore.create(args.kv_store)

    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]

    if len(ctx) == 0:
        ctx = [mx.cpu()]
        logger.warning('Use cpu for train')
    else:
        logger.info('Use %s for train' % ctx )

    if net.startswith('legacy'):
        prefix = net
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, args.begin_epoch)
        (new_sym, new_args) = get_fine_tune_model(
            sym, arg_params, args.num_classes, 'flatten0')
    else:
        new_sym = get_symbol(args)
        aux_params = None
        new_args = None

    args.batch_size = args.batch_size * len(ctx)

    epoch_size = max(int(args.num_examples / args.batch_size / kv.num_workers), args.num_epoch)
    lr_scheduler = multi_factor_scheduler(args.begin_epoch, epoch_size, lr_steps=args.lr_steps)
    optimizer_params = {
            'learning_rate': args.lr,
            'momentum': args.mom,
            'wd': args.wd,
            'lr_scheduler': lr_scheduler}
    initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)

    model = mx.mod.Module(
        context = ctx,
        symbol = new_sym
    )

    train, val = get_dataIter_from_rec(args=args)

    # eval_metric = list()
    # eval_metric.append(mx.metric.np(acc))
    # eval_metric.append(mx.metric.np(loss))

    labels = {
        'Gender' : 2,
        'Mask' : 2,
        'Glass' : 3,
        'MouthOpen' : 3,
        'EyesOpen' : 3
    }
    # add acc metric for every class
    index = 0
    pred_offset = 0
    metric_list = []
    for key in labels:
        metric_list.append(ClassAccMetric(name=key, label_index=index, pred_offset=pred_offset, class_num=labels[key]))
        index += 1
        pred_offset += labels[key]
    eval_metric = mx.metric.CompositeEvalMetric(metric_list)

    def _epoch_callback(epoch, symbol, arg, aux):
        mx.model.save_checkpoint(args.save_result, 1, symbol, arg, aux)
    # checkpoint = mx.callback.do_checkpoint(args.save_result)


    model.fit(train,
            begin_epoch=args.begin_epoch,
            num_epoch=args.num_epoch,
            eval_data=val,
            eval_metric=eval_metric,
            validation_metric=eval_metric,
            kvstore=kv,
            optimizer='sgd',
            optimizer_params=optimizer_params,
            arg_params=new_args,
            aux_params=aux_params,
            initializer=initializer,
            allow_missing=True,
            batch_end_callback=mx.callback.Speedometer(args.batch_size, 20),
            epoch_end_callback=_epoch_callback)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='score a model on a dataset')
    parser.add_argument('--model',         type=str, default= None)
    parser.add_argument('--gpus',          type=str, default= fc.gpus)
    parser.add_argument('--batch-size',    type=int, default= fc.batch_size)
    parser.add_argument('--begin-epoch',         type=int, default=0)
    parser.add_argument('--image-shape',   type=str, default='3,224,224')
    parser.add_argument('--imgrec-train',  type=str, default=fc.imgrec_train)
    parser.add_argument('--imgrec-val',    type=str, default=fc.imgrec_val)
    parser.add_argument('--imgidx-train',  type=str, default=fc.imgidx_train)
    parser.add_argument('--imgidx-val',    type=str, default=fc.imgidx_val)
    parser.add_argument('--num-classes',   type=int, default=fc.num_class)
    parser.add_argument('--lr',            type=float, default=fc.base_lr)
    parser.add_argument('--lr-steps', type=str, default='50, 100, 200', help='steps of lr changing')
    parser.add_argument('--num-epoch',     type=int, default=fc.num_epoch)
    parser.add_argument('--kv-store',      type=str, default='device', help='the kvstore type')
    parser.add_argument('--save-result',   type = str, default=fc.model_path, help='the save path')
    parser.add_argument('--num-examples',  type=int, default=fc.num_examples)
    parser.add_argument('--mom',           type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--wd',            type=float, default=0.0001, help='weight decay for sgd')
    parser.add_argument('--save-name',     type=str, help='the save name of model')
    parser.add_argument('--rand-mirror', type=int, default=True, help='if do random mirror in training')
    parser.add_argument('--cutoff', type=int, default=0, help='cut off aug')
    parser.add_argument('--color', type=int, default=0, help='color jittering aug')
    parser.add_argument('--ce-loss', default=False, action='store_true', help='if output ce loss')

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    kv = mx.kvstore.create(args.kv_store)

    logging.info(args)

    train_model(args=args, net="mobilenet")

