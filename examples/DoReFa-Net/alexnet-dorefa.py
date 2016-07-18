#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: alexnet-dorefa.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import cv2
import tensorflow as tf
import argparse
import numpy as np
import multiprocessing
import msgpack
import os

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

BITW = 1
BITA = 2
BITG = 4

GRAD_DEFINED = False
def get_dorefa(bitW, bitA, bitG):
    """ return the three quantization functions fw, fa, fg, for weights,
    activations and gradients respectively"""
    G = tf.get_default_graph()

    def quantize(x, k):
        n = float(2**k-1)
        with G.gradient_override_map({"Floor": "Identity"}):
            return tf.round(x * n) / n

    def fw(x):
        if bitW == 32:
            return x
        if bitW == 1:   # BWN
            with G.gradient_override_map({"Sign": "Identity"}):
                E = tf.stop_gradient(tf.reduce_mean(tf.abs(x)))
                return tf.sign(x / E) * E
        x = tf.tanh(x)
        x = x / tf.reduce_max(tf.abs(x)) * 0.5 + 0.5
        return 2 * quantize(x, bitW) - 1

    def fa(x):
        if bitA == 32:
            return x
        return quantize(x, bitA)

    global GRAD_DEFINED
    if not GRAD_DEFINED:
        @tf.RegisterGradient("FGGrad")
        def grad_fg(op, x):
            rank = x.get_shape().ndims
            assert rank is not None
            maxx = tf.reduce_max(tf.abs(x), list(range(1,rank)), keep_dims=True)
            x = x / maxx
            n = float(2**bitG-1)
            x = x * 0.5 + 0.5 + tf.random_uniform(
                    tf.shape(x), minval=-0.5/n, maxval=0.5/n)
            x = tf.clip_by_value(x, 0.0, 1.0)
            x = quantize(x, bitG) - 0.5
            return x * maxx * 2
    GRAD_DEFINED = True

    def fg(x):
        if bitG == 32:
            return x
        with G.gradient_override_map({"Identity": "FGGrad"}):
            return tf.identity(x)
    return fw, fa, fg

class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, 224, 224, 3], 'input'),
                InputVar(tf.int32, [None], 'label') ]

    def _build_graph(self, input_vars, is_training):
        image, label = input_vars
        image = image / 255.0

        fw, fa, fg = get_dorefa(BITW, BITA, BITG)
        # monkey-patch tf.get_variable to apply fw
        old_get_variable = tf.get_variable
        def new_get_variable(name, shape=None, **kwargs):
            v = old_get_variable(name, shape, **kwargs)
            # don't binarize first and last layer
            if name != 'W' or 'conv0' in v.op.name or 'fct' in v.op.name:
                return v
            else:
                logger.info("Binarizing weight {}".format(v.op.name))
                return fw(v)
        tf.get_variable = new_get_variable

        def nonlin(x):
            if BITA == 32:
                return tf.nn.relu(x)
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
            return fa(nonlin(x))

        with argscope(BatchNorm, decay=0.9, epsilon=1e-4, use_local_stat=is_training), \
                argscope([Conv2D, FullyConnected], use_bias=False, nl=tf.identity):
            logits = (LinearWrap(image)
                .Conv2D('conv0', 96, 12, stride=4, padding='VALID')
                .apply(activate)

                .Conv2D('conv1', 256, 5, padding='SAME', split=2)
                .apply(fg)
                .BatchNorm('bn1')
                .MaxPooling('pool1', 3, 2, padding='SAME')
                .apply(activate)

                .Conv2D('conv2', 384, 3)
                .apply(fg)
                .BatchNorm('bn2')
                .MaxPooling('pool2', 3, 2, padding='SAME')
                .apply(activate)

                .Conv2D('conv3', 384, 3, split=2)
                .apply(fg)
                .BatchNorm('bn3')
                .apply(activate)

                .Conv2D('conv4', 256, 3, split=2)
                .apply(fg)
                .BatchNorm('bn4')
                .MaxPooling('pool4', 3, 2, padding='VALID')
                .apply(activate)

                .tf.nn.dropout(0.5 if is_training else 1.0)
                .FullyConnected('fc0', 4096)
                .apply(fg)
                .BatchNorm('bnfc0')
                .apply(activate)

                .tf.nn.dropout(0.5 if is_training else 1.0)
                .FullyConnected('fc1', 4096)
                .apply(fg)
                .BatchNorm('bnfc1')
                .apply(nonlin)
                .FullyConnected('fct', 1000)())
        tf.get_variable = old_get_variable


        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label, 1)
        nr_wrong = tf.reduce_sum(wrong, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train_error_top1'))
        wrong = prediction_incorrect(logits, label, 5)
        nr_wrong = tf.reduce_sum(wrong, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train_error_top5'))

        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(5e-6))
        add_moving_summary(cost, wd_cost)

        add_param_summary([('.*/W', ['histogram', 'rms'])])
        self.cost = tf.add_n([cost, wd_cost], name='cost')

def get_data(dataset_name):
    isTrain = dataset_name == 'train'
    #ds = dataset.ILSVRC12('/home/wyx/data/fake_ilsvrc', dataset_name, shuffle=True if isTrain else False)
    ds = dataset.ILSVRC12(args.data, dataset_name,
            shuffle=True if isTrain else False,
            dir_structure='train')

    meta = dataset.ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    pp_mean_224 = pp_mean[16:-16,16:-16,:]

    if isTrain:
        class Resize(imgaug.ImageAugmentor):
            def __init__(self):
                self._init(locals())
            def _augment(self, img, _):
                h, w = img.shape[:2]
                size = 224
                scale = self.rng.randint(size, 308) * 1.0 / min(h, w)
                scaleX = scale * self.rng.uniform(0.85, 1.15)
                scaleY = scale * self.rng.uniform(0.85, 1.15)
                desSize = map(int, (max(size, min(w, scaleX * w)),\
                    max(size, min(h, scaleY * h))))
                dst = cv2.resize(img, tuple(desSize),
                     interpolation=cv2.INTER_CUBIC)
                return dst

        augmentors = [
            Resize(),
            imgaug.Rotation(max_deg=10),
            imgaug.RandomApplyAug(imgaug.GaussianBlur(3), 0.5),
            imgaug.Brightness(30, True),
            imgaug.Gamma(),
            imgaug.Contrast((0.8,1.2), True),
            imgaug.RandomCrop((224, 224)),
            imgaug.RandomApplyAug(imgaug.JpegNoise(), 0.8),
            imgaug.RandomApplyAug(imgaug.GaussianDeform(
                [(0.2, 0.2), (0.2, 0.8), (0.8,0.8), (0.8,0.2)],
                (224, 224), 0.2, 3), 0.1),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean_224),
        ]
    else:
        def resize_func(im):
            h, w = im.shape[:2]
            scale = 256.0 / min(h, w)
            desSize = map(int, (max(224, min(w, scale * w)),\
                                max(224, min(h, scale * h))))
            im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
            return im
        augmentors = [
            imgaug.MapImage(resize_func),
            imgaug.CenterCrop((224, 224)),
            imgaug.MapImage(lambda x: x - pp_mean_224),
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, 32, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(12, multiprocessing.cpu_count()))
    return ds

def get_config():
    logger.auto_set_dir()

    # prepare dataset
    data_train = get_data('train')
    data_test = get_data('val')

    lr = tf.Variable(1e-4, trainable=False, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=data_train,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-5),
        callbacks=Callbacks([
            StatPrinter(),
            ModelSaver(),
            HumanHyperParamSetter('learning_rate'),
            InferenceRunner(data_test,
                [ScalarStats('cost'),
                 ClassificationError('wrong-top1', 'val-top1-error'),
                 ClassificationError('wrong-top5', 'val-top5-error')])
        ]),
        model=Model(),
        step_per_epoch=3000,
        max_epoch=200,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the GPU to use') # nargs='*' in multi mode
    parser.add_argument('--load', help='load a checkpoint')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--dorefa',
            help='number of bits for W,A,G, separated by comma. Defaults to \'1,2,4\'',
            default='1,2,4')
    args = parser.parse_args()

    BITW, BITA, BITG = map(int, args.dorefa.split(','))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    SyncMultiGPUTrainer(config).train()
