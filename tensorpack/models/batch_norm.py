#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: batch_norm.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from copy import copy
import re

from ..utils import logger, EXTRA_SAVE_VARS_KEY
from ._common import layer_register

__all__ = ['BatchNorm']

# http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
# TF batch_norm only works for 4D tensor right now: #804
# decay: being too close to 1 leads to slow start-up. torch use 0.9.
# eps: torch: 1e-5. Lasagne: 1e-4
@layer_register(log_shape=False)
def BatchNorm(x, use_local_stat=True, decay=0.9, epsilon=1e-5):
    """
    Batch normalization layer as described in:

    `Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariance Shift <http://arxiv.org/abs/1502.03167>`_.

    Notes:

    * Whole-population mean/variance is calculated by a running-average mean/variance.
    * Epsilon for variance is set to 1e-5, as is `torch/nn <https://github.com/torch/nn/blob/master/BatchNormalization.lua>`_.

    :param input: a NHWC tensor or a NC vector
    :param use_local_stat: bool. whether to use mean/var of this batch or the moving average.  Set to True in training and False in testing
    :param decay: decay rate. default to 0.999.
    :param epsilon: default to 1e-5.
    """

    shape = x.get_shape().as_list()
    assert len(shape) in [2, 4]

    n_out = shape[-1]  # channel
    assert n_out is not None
    beta = tf.get_variable('beta', [n_out])
    gamma = tf.get_variable('gamma', [n_out],
        initializer=tf.ones_initializer)

    if len(shape) == 2:
        batch_mean, batch_var = tf.nn.moments(x, [0], keep_dims=False)
    else:
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims=False)
    # just to make a clear name.
    batch_mean = tf.identity(batch_mean, 'mean')
    batch_var = tf.identity(batch_var, 'variance')

    # XXX a hack to handle training tower & prediction tower together....
    emaname = 'EMA'
    if not batch_mean.name.startswith('towerp'):
        with tf.name_scope(None): # https://github.com/tensorflow/tensorflow/issues/2740
            ema = tf.train.ExponentialMovingAverage(decay=decay, name=emaname)
            ema_apply_op = ema.apply([batch_mean, batch_var])
            ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
            tf.add_to_collection(EXTRA_SAVE_VARS_KEY, ema_mean)
            tf.add_to_collection(EXTRA_SAVE_VARS_KEY, ema_var)
    else:
        # use training-statistics in prediction
        assert not use_local_stat
        with tf.name_scope(None):
            ema = tf.train.ExponentialMovingAverage(decay=decay, name=emaname)
            mean_var_name = ema.average_name(batch_mean) + ':0'
            var_var_name = ema.average_name(batch_var) + ':0'

        G = tf.get_default_graph()
        # find training statistics in training tower
        try:
            mean_name = re.sub('towerp[0-9]+/', '', mean_var_name)
            var_name = re.sub('towerp[0-9]+/', '', var_var_name)
            #var_name = batch_var.op.name[prefixlen:] + '/' + emaname + ':0'
            ema_mean = G.get_tensor_by_name(mean_name)
            ema_var = G.get_tensor_by_name(var_name)
        except KeyError:
            mean_name = re.sub('towerp[0-9]+/', 'tower0/', mean_var_name)
            var_name = re.sub('towerp[0-9]+/', 'tower0/', var_var_name)
            ema_mean = G.get_tensor_by_name(mean_name)
            ema_var = G.get_tensor_by_name(var_name)
        #logger.info("In prediction, using {} instead of {} for {}".format(
            #mean_name, ema_mean.name, batch_mean.name))

    if use_local_stat:
        with tf.control_dependencies([ema_apply_op]):
            batch = tf.cast(tf.shape(x)[0], tf.float32)
            mul = tf.select(tf.equal(batch, 1.0), 1.0, batch / (batch - 1))
            batch_var = batch_var * mul  # use unbiased variance estimator in training
            return tf.nn.batch_normalization(
                x, batch_mean, batch_var, beta, gamma, epsilon, 'bn')
    else:
        return tf.nn.batch_normalization(
            x, ema_mean, ema_var, beta, gamma, epsilon, 'bn')
