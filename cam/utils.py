

import tensorflow as tf


def _print_shape_py(t, msg):
  print("%s shape: %s" % (msg, t.shape))
  return sum(t.shape)


def print_shape(t, msg="", dtype=tf.float32):
  add = tf.py_func(_print_shape_py, [t, msg], tf.int64)
  shp = t.get_shape()
  ret = t + tf.cast(tf.reduce_max(add) - tf.reduce_max(add), dtype=dtype)
  ret.set_shape(shp)
  return ret


def _print_data_py(t, msg):
  print("%s shape: %s" % (msg, t.shape))
  print("%s data: %s" % (msg, t))
  return sum(t.shape)


def print_data(t, msg="", dtype=tf.float32):
  add = tf.py_func(_print_data_py, [t, msg], tf.int64)
  shp = t.get_shape()
  ret = t + tf.cast(tf.reduce_max(add) - tf.reduce_max(add), dtype=dtype)
  ret.set_shape(shp)
  return ret


def get_sequence_length(t):
  # t shape: batch_size, max_sequence_length, embed_size
  not_padding = tf.greater(tf.reduce_sum(tf.abs(t), axis=2), 0.000001)
  not_padding_with_guardian = tf.pad(not_padding, [[0, 0], [0, 1]])
  indices = tf.where(tf.logical_not(not_padding_with_guardian))
  length = tf.segment_min(indices[:, 1], indices[:, 0])
  return tf.cast(length, tf.int32)
  

