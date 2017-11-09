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


def get_length_from_raw(raw_x):
  """Extracts sequence lengths from the raw feature.

  Example:
    raw_x = [
      [123, 3, 2, 0, 0],
      [321, 1, 0, 0, 0]]
    return:
      [3, 2]

  Args:
    raw_x: A [batch_size, max_length] int32 tensor
  
  Returns:
    A [batch_size] int32 tensor with sequence lengths
  """
  not_padding = tf.not_equal(raw_x, text_encoder.PAD_ID)
  not_padding_with_guardian = tf.pad(not_padding, [[0, 0], [0, 1]])
  indices = tf.where(tf.logical_not(not_padding_with_guardian))
  length = tf.segment_min(indices[:, 1], indices[:, 0])
  return length
  

def gather_2d(params, indices):
  """This is a batched version of tf.gather(), ie. it applies tf.gather() to
  each batch separately.

  Example:
    params = [[10, 11, 12, 13, 14],
              [20, 21, 22, 23, 24]]
    indices = [[0, 0, 1, 1, 1, 2],
               [1, 3, 0, 0, 2, 2]]
    result = [[10, 10, 11, 11, 11, 12],
              [21, 23, 20, 20, 22, 22]]

  Args:
    params: A [batch_size, n, ...] tensor with data
    indices: A [batch_size, num_indices] int32 tensor with indices into params.
             Entries must be smaller than n

  Returns:
    The result of tf.gather() on each entry of the batch.
  """
  # TODO(fstahlberg): Curse TF for making this so awkward.
  batch_size = tf.shape(params)[0]
  num_indices = tf.shape(indices)[1]
  batch_indices = tf.tile(tf.expand_dims(tf.range(batch_size), 1), 
                          [1, num_indices])
  # batch_indices is [[0,0,0,0,...],[1,1,1,1,...],...]
  gather_nd_indices = tf.stack([batch_indices, indices], axis=2)
  return tf.gather_nd(params, gather_nd_indices)


def expand_memory_by_pop(is_pop, memory, offset=1):
  """Expands a memory according POP signals.

  This method is useful when the POP attention mechanism is used. It
  repeats entries in the memory such that we can walk through the
  expanded memory one step at a time and obtain the correct attention.

  Example (symbolic, without batch size):
    is_pop = [False, True, False, False, True]
    memory = [a, b, c]
    offset = 0
    result = [a, b, b, b, c]

    is_pop = [False, True, False, False]
    memory = [a, b, c, d]
    offset = 1
    result = [a, a, b, b, b]

  Args:
    is_pop: A [batch_size, max_seq_length] bool tensor with POP signals
      The number of True values in each line must be lower than memory_length
    memory: A [batch_size, memory_length, ...] tensor
    offset (int): Padding width on the left. This defaults to 1 since we
      usually want to update the attention *after* POP is produced.

  Returns:
    A [batch_size, max_seq_length+offset, ...] tensor with entries 
    from memory, repeated according the POP signals.
  """
  indices = tf.cumsum(tf.cast(is_pop, tf.int32), axis=1)
  if offset:
    indices = tf.pad(indices, [[0, 0], [offset, 0]])
  return gather_2d(memory, indices)


def expand_memory_by_pop_1d(is_pop, memory, offset=1):
  """1D version of expand_memory_by_pop.

  Args:
    is_pop: A [max_seq_length,] bool tensor with POP signals
    memory: A [ memory_length, ...] tensor
    offset (int): Padding width on the left.

  Returns:
    A [max_seq_length+offset, ...] tensor with entries 
    from memory, repeated according the POP signals.
  """
  indices = tf.cumsum(tf.cast(is_pop, tf.int32))
  if offset:
    indices = tf.pad(indices, [[offset, 0]])
  return tf.gather(memory, indices)


def _get_string_to_id(encoder):
  """Get a dictionary from string representations to token ID for
  the encoder. This is a hack and only works for subword and token encoders.
  """
  try:
    return encoder._subtoken_string_to_id
  except AttributeError:
    try:
      return encoder._token_to_id
    except AttributeError as e:
      tf.logging.fatal("Could not read encoder vocabulary. Please either "
                        "use the Subword or Token encoder or specify "
                        "reserved IDs directly")
      raise e
  

def look_up_token_id(encoder, key, token_str, hparams):
  if not hasattr(hparams, key):
    try:
      token_id = _get_string_to_id(encoder)[token_str]
    except KeyError as e:
      tf.logging.fatal("%s could not be found in the target vocabulary! "
                      "Please either add %s to the vocabulary file or "
                      "specify the ID directly with hparams.%s"
                      % (token_str, token_str, key))
      raise e
    tf.logging.info("Use ID %d for pop signal %s", token_id, token_str)
    hparams.add_hparam(key, token_id)


def extract_max_terminal_id(encoder, hparams):
  if not hasattr(hparams, 'max_terminal_id'):
    string_to_id = _get_string_to_id(encoder)
    max_terminal_id = 0
    min_nonterminal_id = 1000000
    for s, i in string_to_id.iteritems():
      if s[:2] == "##" and s[-2:] == "##":
        min_nonterminal_id = min(min_nonterminal_id, i)
      else:
        max_terminal_id = max(max_terminal_id, i)
    if min_nonterminal_id != max_terminal_id + 1:
      tf.logging.warn("Overlapping non-terminal and terminal ID "
                      "ranges: min nonterminal is %d" % min_nonterminal_id)
    tf.logging.info("Set maximum terminal ID to %d", max_terminal_id)
    hparams.add_hparam("max_terminal_id", max_terminal_id)

