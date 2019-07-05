# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers

from usr import utils as usr_utils

import tensorflow as tf
import numpy as np


def transform_qkv(query_antecedent, key_antecedent, value_antecedent, total_key_depth,
                total_value_depth, q_filter_width=1, kv_filter_width=1,
                q_padding="VALID", kv_padding="VALID"):
  """Computes query, key and value.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    total_key_depth: an integer
    total_value_depth: and integer
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
    to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.

  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  q = common_layers.conv1d(
      query_antecedent, total_key_depth, q_filter_width, padding=q_padding,
      name="q_transform")
  k = common_layers.conv1d(
      key_antecedent,
      total_key_depth,
      1,
      padding=kv_padding,
      name="k_transform")
  v = common_layers.conv1d(
      value_antecedent,
      total_key_depth,
      1,
      padding=kv_padding,
      name="v_transform")
  return q, k, v


def multihead_attention_qkv(query_antecedent,
                        key_antecedent,
                        value_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        max_relative_position=None,
                        image_shapes=None,
                        attention_type="dot_product",
                        block_length=128,
                        block_width=128,
                        q_filter_width=1,
                        kv_filter_width=1,
                        q_padding="VALID",
                        kv_padding="VALID",
                        cache=None,
                        gap_size=0,
                        num_memory_blocks=2,
                        attention_order=1,
                        name=None,
                        **kwargs):
  """Multihead scaled-dot-product attention with separate key and value inputs
  rather than a single memory input.input/output transformations.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    ...
    attention_order (int): For high order attention like dot_product_highorder
    (rest: see common_attention.multihead_attention)
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  with tf.variable_scope(
      name,
      default_name="multihead_attention",
      values=[query_antecedent, key_antecedent, value_antecedent]):
    if value_antecedent is None:
      q, k, v = common_attention.compute_qkv(
                          query_antecedent, key_antecedent, total_key_depth,
                          total_value_depth, q_filter_width, kv_filter_width,
                          q_padding, kv_padding)
    else:
      q, k, v = transform_qkv(query_antecedent, key_antecedent, value_antecedent, total_key_depth,
                          total_value_depth, q_filter_width, kv_filter_width,
                          q_padding, kv_padding)

    if cache is not None:
      if attention_type != "dot_product":
        raise NotImplementedError(
            "Caching is not guaranteed to work with attention types other than"
            " dot_product.")
      if bias is None:
        raise ValueError("Bias required for caching. See function docstring "
                         "for details.")
      k = cache["k"] = tf.concat([cache["k"], k], axis=1)
      v = cache["v"] = tf.concat([cache["v"], v], axis=1)

    q = common_attention.split_heads(q, num_heads)
    k = common_attention.split_heads(k, num_heads)
    v = common_attention.split_heads(v, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head**-0.5

    if "," in attention_type:
      num_types = attention_type.count(",") + 1
      qs = tf.split(q, num_types, axis=1)
      ks = tf.split(k, num_types, axis=1)
      vs = tf.split(v, num_types, axis=1)
      key_depth_per_head = total_key_depth // num_heads // num_types
    else:
      qs = [q]
      ks = [k]
      vs = [v]
      key_depth_per_head = total_key_depth // num_heads
    additional_returned_value = None
    xs = []
    for q, k, v, att_type in zip(qs, ks, vs, attention_type.split(",")):
      q *= key_depth_per_head**-0.5
      if callable(att_type):  # Generic way to extend multihead_attention
        x = att_type(q, k, v, **kwargs)
        if isinstance(x, tuple):
          x, additional_returned_value = x  # Unpack
      elif att_type == "dot_product":
        x = common_attention.dot_product_attention(q, k, v, bias, dropout_rate, image_shapes)
      elif att_type == "dot_product_highorder":
        x = dot_product_highorder_attention(q, k, v, bias, dropout_rate, image_shapes, attention_order=attention_order)
      elif att_type == "dot_product_highorder_shared":
        x = dot_product_highorder_shared_attention(q, k, v, bias, dropout_rate, image_shapes, attention_order=attention_order)
      elif att_type == "dot_product_relative":
        x = common_attention.dot_product_attention_relative(q, k, v, bias, max_relative_position,
                                           dropout_rate, image_shapes)
      elif att_type == "local_mask_right":
        x = common_attention.masked_local_attention_1d(q, k, v, block_length=block_length)
      elif att_type == "local_unmasked":
        x = common_attention.local_attention_1d(
            q, k, v, block_length=block_length, filter_width=block_width)
      elif att_type == "masked_dilated_1d":
        x = common_attention.masked_dilated_self_attention_1d(q, k, v, block_length,
                                             block_width,
                                             gap_size,
                                             num_memory_blocks)
      else:
        assert att_type == "unmasked_dilated_1d"
        x = common_attention.dilated_self_attention_1d(q, k, v, block_length,
                                      block_width,
                                      gap_size,
                                      num_memory_blocks)
      xs.append(x)
    x = xs[0] if len(xs) == 1 else tf.concat(xs, axis=1)
    x = common_attention.combine_heads(x)
    x = common_layers.conv1d(x, output_depth, 1, name="output_transform")
    if additional_returned_value is not None:
      return x, additional_returned_value
    return x


def dot_product_highorder_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          image_shapes=None,
                          attention_order=2,
                          name=None,
                          make_image_summary=True):
  """High order dot-product attention. Attention is applied repeatedly
  to generate query vectors. For example, 2-order attention uses q,k,v
  to generate a new query vector q'. The final attention result is
  computed with q',k,v.

  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    image_shapes: optional tuple of integer scalars.
      see comments for attention_image_summary()
    attention_order (int): Attention order (number of steps)
    name: an optional string
    make_image_summary: True if you want an image summary.

  Returns:
    A Tensor.
  """
  if attention_order == 1:
    return common_attention.dot_product_attention(q, k, v, bias, dropout_rate, image_shapes,
        name=name, make_image_summary=make_image_summary)
  # Split q, k in attention_order pieces
  qs = tf.split(q, attention_order, axis=3)
  ks = tf.split(k, attention_order, axis=3)
  with tf.variable_scope(
      name, default_name="dot_product_highorder_attention", values=[q, k, v]):
    for idx in xrange(attention_order):
      # [batch, num_heads, query_length, memory_length]
      q = tf.matmul(weights, qs[idx]) if idx != 0 else qs[0]
      logits = tf.matmul(q, ks[idx], transpose_b=True)
      if bias is not None:
        logits += bias
      weights = tf.nn.softmax(logits, name="attention_weights")
    # dropping out the attention links for each of the heads
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    if (not tf.get_variable_scope().reuse and
        # Summaries don't work well within tf.while_loop()
        "/while/" not in tf.contrib.framework.get_name_scope() and
        make_image_summary):
      common_attention.attention_image_summary(weights, image_shapes)
    return tf.matmul(weights, v)


def dot_product_highorder_shared_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          image_shapes=None,
                          attention_order=2,
                          name=None,
                          make_image_summary=True):
  """High order dot-product attention. Attention is applied repeatedly
  to generate query vectors. For example, 2-order attention uses q,k,v
  to generate a new query vector q'. The final attention result is
  computed with q',k,v.

  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    image_shapes: optional tuple of integer scalars.
      see comments for attention_image_summary()
    attention_order (int): Attention order (number of steps)
    name: an optional string
    make_image_summary: True if you want an image summary.

  Returns:
    A Tensor.
  """
  if attention_order == 1:
    return common_attention.dot_product_attention(q, k, v, bias, dropout_rate, image_shapes,
        name=name, make_image_summary=make_image_summary)
  with tf.variable_scope(
      name, default_name="dot_product_highorder_shared_attention", values=[q, k, v]):
    for _ in xrange(attention_order):
      # [batch, num_heads, query_length, memory_length]
      logits = tf.matmul(q, k, transpose_b=True)
      if bias is not None:
        logits += bias
      weights = tf.nn.softmax(logits, name="attention_weights")
      q = tf.matmul(weights, v)
    # dropping out the attention links for each of the heads
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    if (not tf.get_variable_scope().reuse and
        # Summaries don't work well within tf.while_loop()
        "/while/" not in tf.contrib.framework.get_name_scope() and
        make_image_summary):
      common_attention.attention_image_summary(weights, image_shapes)
    return tf.matmul(weights, v)


def dot_product_osm_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          name=None):
  """dot-product attention.

  Args:
    q: a Tensor with shape [batch, heads, length (time), depth]
    k: a Tensor with shape [batch, heads, length (time), length (annotation), depth]
    v: a Tensor with shape [batch, heads, length, depth]
    bias: [1, 1, length (time), length (annotation)] bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    name: an optional string

  Returns:
    A [batch, heads, length, depth] float32 Tensor.
  """
  with tf.variable_scope(
      name, default_name="dot_product_osm_attention", values=[q, k, v]) as scope:
    logits = tf.matmul(tf.expand_dims(q, 3), k, transpose_b=True)
    # logits is [batch, heads, length (time), 1, length (annotation)]
    logits = tf.squeeze(logits, axis=3)
    if bias is not None:
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    # dropping out the attention links for each of the heads
    weights = tf.nn.dropout(weights, 1.0 - dropout_rate)
    # logits and weights are [batch, heads, length, length]
    return tf.matmul(weights, v)


def decode_osm_py(labels):
  EOP = 4
  GAP = 5
  JUMP_FWD = 6
  JUMP_BWD = 7
  def compile_ops_jump(compiled, head, step, ops):
    head += step
    try:
      while ops[compiled[head]] != GAP:
        head += step
    except IndexError:
      pass # This is a jump to the end, so it's fine
    return head

  def compile_ops_insert(compiled, head, op):
    compiled = compiled[:head] + [op] + compiled[head:]
    head += 1
    return head, compiled


  batch_size = labels.shape[0]
  length = labels.shape[1]
  pos_matrices = []
  for b in xrange(batch_size):
    ops = [op for op in labels[b, :] if op > 1] # Remove EOS and PAD
    compiled = []
    head = 0
    pos_rows = [np.zeros((length,), dtype=labels.dtype)] # For shifting
    for op_idx, op in enumerate(ops):
      if op == JUMP_FWD:
        head = compile_ops_jump(compiled, head, 1, ops)
      elif op == JUMP_BWD:
        head = compile_ops_jump(compiled, head, -1, ops)
      head, compiled = compile_ops_insert(compiled, head, op_idx)
      pos_row = np.zeros((length,), dtype=labels.dtype)
      for pos, c in enumerate(compiled):
        pos_row[c+1] = pos+1 # Shift by one as in trasnformer decoder input
      pos_rows.append(pos_row)
    if len(pos_rows) < length: # padding
      pos_rows.extend([np.zeros((length,), dtype=labels.dtype)] * (length - len(pos_rows)))
    pos_matrix = np.stack(pos_rows, axis=0)
    pos_matrices.append(pos_matrix)
  return np.stack(pos_matrices, axis=0)
  #return np.ones((batch_size, length, length), dtype=labels.dtype)


def decode_osm(labels):
  """Get the positions in the compiled sequences at each timestep.

  Args:
    labels: An [batch, length] int32 tensor with the labels.

  Returns:
    An [batch, length (time), length (annotation)] int32 tensor in which the
    [b, i, j]th entry stores the position of the j-th OSM operation in the 
    compiled string after i time steps in batch b.
  """
  shp = labels.get_shape()
  ret = tf.py_func(decode_osm_py, [labels], tf.int32)
  ret.set_shape([shp[0], shp[1], shp[1]])
  return ret
  #shp = tf.shape(labels)
  #return tf.random_uniform((shp[0], shp[1], shp[1]), maxval=13, dtype=tf.int32)
  

def compute_qkv_osm(query_antecedent,
                query_antecedent_raw,
                total_key_depth,
                total_value_depth,
                q_filter_width=1,
                kv_filter_width=1,
                q_padding="VALID",
                kv_padding="VALID"):
  """Computes query, key and value.

  Args:
    query_antecedent: a Tensor with shape [batch, length, channels]
    query_antecedent_raw: a int Tensor with shape [batch, length]
    total_key_depth: an integer
    total_value_depth: and integer
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
    to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.

  Returns:
    q, k, v : k have shape [batch, length, length depth], q,v has shape
    [batch, length, depth]
  """
  def _compute(inp, depth, filter_width, padding, name):
    if filter_width == 1:
      return tf.layers.dense(inp, depth, use_bias=False, name=name)
    else:
      return common_layers.conv1d(inp, depth, filter_width, padding, name=name)

  positions_raw = decode_osm(query_antecedent_raw)
  #positions_raw is [batch_size, length, length]
  #embed_lookup = common_attention.get_timing_signal_1d(tf.reduce_max(positions_raw)+1, total_key_depth)
  embed_lookup = common_attention.get_timing_signal_1d(tf.reduce_max(positions_raw)+1, total_key_depth // 2, 1.0, 5.0e3)
  embed_lookup = tf.squeeze(embed_lookup, axis=0)
  positions_embed = tf.gather(embed_lookup, positions_raw)
  #positions_embed is [batch_size, length (time), length (annotation), total_key_depth]
  shp = tf.shape(query_antecedent)

  combined_antecedent = tf.concat(
      [tf.tile(tf.expand_dims(query_antecedent, 1), (1, shp[1], 1, 1)),
       positions_embed], axis=3)
  #combined_antecedent = tf.tile(tf.expand_dims(query_antecedent, 1), (1, shp[1], 1, 1)) + positions_embed

  # combined_antecedent is [batch_size, length (time), length (annotation), combined_key_depth]
  # For q we only need annotation i at time i, ie. we can save one dimension
  q_combined_antecedent = tf.matrix_diag_part(tf.transpose(combined_antecedent, [0, 3, 1, 2]))
  # q_combined_antecedent is [batch, 2*total_key_depth, length]
  
  q = _compute(
      tf.transpose(q_combined_antecedent, [0, 2, 1]), 
      total_key_depth, q_filter_width, q_padding, "q")
  k = _compute(
      combined_antecedent, total_key_depth, kv_filter_width, kv_padding, "k")
  v = _compute(
      query_antecedent, total_value_depth, kv_filter_width, kv_padding, "v")
  return q, k, v

def split_heads_5d(x, num_heads):
  return tf.transpose(common_attention.split_last_dimension(x, num_heads), [0, 3, 1, 2, 4])


def multihead_attention_osm(query_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        max_relative_position=None,
                        attention_type="dot_product",
                        block_length=128,
                        block_width=128,
                        q_filter_width=1,
                        kv_filter_width=1,
                        q_padding="VALID",
                        kv_padding="VALID",
                        cache=None,
                        gap_size=0,
                        num_memory_blocks=2,
                        name=None,
                        query_antecedent_raw=None,
                        **kwargs):
  """Multihead scaled-dot-product attention with separate key and value inputs
  rather than a single memory input.input/output transformations.

  Args:
    query_antecedent: a Tensor with shape [batch, length, channels]
    bias: [1, 1, length, length] bias Tensor (see attention_bias())
    ...
    query_antecedent_raw: a int32 Tensor with shape [batch, length]
    (rest: see common_attention.multihead_attention)
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  with tf.variable_scope(
      name,
      default_name="multihead_attention",
      values=[query_antecedent]):
    q, k, v = compute_qkv_osm(
                          query_antecedent, query_antecedent_raw, total_key_depth,
                          total_value_depth, q_filter_width, kv_filter_width,
                          q_padding, kv_padding)
    q = common_attention.split_heads(q, num_heads)
    k = split_heads_5d(k, num_heads)
    v = common_attention.split_heads(v, num_heads)
    # k has shape [batch, heads, length (time), length (annotaion), total_[key|value]_depth // num_heads]
    # q,v have shape [batch, heads, length, total_[key|value]_depth // num_heads]
    key_depth_per_head = total_key_depth // num_heads

    q *= key_depth_per_head**-0.5
    x = dot_product_osm_attention(q, k, v, bias, dropout_rate)
    x = common_attention.combine_heads(x)
    x = tf.layers.dense(
        x, output_depth, use_bias=False, name="output_transform")
    return x

