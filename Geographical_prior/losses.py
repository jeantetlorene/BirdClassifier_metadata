# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 10:09:30 2022

@author:  Copyright 2021 Fagner Cunha

from github https://github.com/alcunha/geo_prior_tf/blob/master/geo_prior/dataloader.py
"""

import tensorflow as tf

def weighted_binary_cross_entropy(pos_weight = 1, epsilon=1e-5):
  def _log(value):
    return (-1)*(tf.math.log(value + epsilon))

  def _call(y_true, y_pred):
    log_loss = pos_weight * y_true * _log(y_pred) \
               + (1 - y_true) * _log(1 - y_pred)

    return tf.reduce_mean(log_loss, axis=-1)

  return _call

def log_loss(epsilon=1e-5):
  def _log(value):
    return (-1)*(tf.math.log(value + epsilon))

  def _call(y_true, y_pred):
    _log_loss = y_true * _log(y_pred)

    return tf.reduce_mean(_log_loss, axis=-1)

  return _call