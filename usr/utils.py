"""Helper functions used throughout this package."""

import tensorflow as tf
from tensor2tensor.layers import common_layers
from tensor2tensor.data_generators import text_encoder


def _print_shape_py(t, msg):
  print("%s shape: %s" % (msg, t.shape))
  return sum(t.shape)


def print_shape(t, msg="", dtype=tf.float32):
  """Print shape of the tensor for debugging."""
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
  """Print shape and content of the tensor for debugging."""
  add = tf.py_func(_print_data_py, [t, msg], tf.int64)
  shp = t.get_shape()
  ret = t + tf.cast(tf.reduce_max(add) - tf.reduce_max(add), dtype=dtype)
  ret.set_shape(shp)
  return ret


def get_feature_with_length(features, name):
  """Reads out embeddings and sequence lengths for a symbol modality 
  from the features.

  Args:
    features (dict): Dictionary with features.
    name (string): Feature to extract (will read features[name] and maybe
                    features[raw_name]

  Returns:
    Pair of (embed, length) tensors, where `embed` is a (batch_size,
    max_len, embed_size) float32 tensor with embeddings, and `length`
    is a (batch_size,) int32 tensor with sequence lengths.
  """
  # features[name] shape: (batch_size, max_len, 1, embed_size)
  embed = common_layers.flatten4d3d(features[name])
  # embed shape: (batch_size, max_len, embed_size)
  if "raw_%s" % name in features:
    raw = tf.squeeze(features["raw_%s" % name], axis=[2, 3])
    not_padding = tf.not_equal(raw, text_encoder.PAD_ID)
  else:
    tf.logging.warn(
      "Feature %s is not exposed by T2T in raw form which makes it difficult "
      "to extract sequence lengths. Consider using the T2T fork at "
      "https://github.com/fstahlberg/tensor2tensor. For now we back off to a "
      "more ineffective way to get sequence lengths to maintain compatibility "
      "with the T2T master fork.", name)
    not_padding = tf.greater(tf.reduce_sum(tf.abs(embed), axis=2), 0.000001)
  not_padding_with_guardian = tf.pad(not_padding, [[0, 0], [0, 1]])
  indices = tf.where(tf.logical_not(not_padding_with_guardian))
  length = tf.segment_min(indices[:, 1], indices[:, 0])
  return embed, tf.cast(length, tf.int32)
  

