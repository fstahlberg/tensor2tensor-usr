# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""transformer (attention).

encoder: [Self-Attention, Feed-forward] x n
decoder: [Self-Attention, Source-Target-Attention, Feed-forward] x n
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

from usr.models.transformer_ext import model_helper
from usr import utils as usr_utils

import tensorflow as tf
import numpy as np


@registry.register_model
class TransformerExt(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  def encode(self, inputs, raw_inputs, target_space, hparams):
    """Encode transformer inputs.

    Args:
      inputs: Transformer inputs [batch_size, input_length, hidden_dim]
      target_space: scalar, target space ID.
      hparams: hyperparmeters for model.

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encodre-decoder attention. [batch_size, input_length]
    """
    inputs = common_layers.flatten4d3d(inputs)

    encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
        transformer_prepare_encoder(inputs, target_space, hparams))

    encoder_input = tf.nn.dropout(
        encoder_input, 1.0 - hparams.layer_prepostprocess_dropout)

    encoder_output = transformer_encoder(
        encoder_input,
        raw_inputs,
        self_attention_bias,
        hparams)

    return encoder_output, encoder_decoder_attention_bias

  def decode(
      self,
      decoder_input,
      encoder_output,
      raw_targets,
      encoder_decoder_attention_bias,
      decoder_self_attention_bias,
      hparams,
      cache=None):
    """Decode Transformer outputs from encoder representation.

    Args:
      decoder_input: inputs to bottom of the model.
          [batch_size, decoder_length, hidden_dim]
      encoder_output: Encoder representation.
          [batch_size, input_length, hidden_dim]
      encoder_decoder_attention_bias: Bias and mask weights for
          encoder-decoder attention. [batch_size, input_length]
      decoder_self_attention_bias: Bias and mask weights for decoder
          self-attention. [batch_size, decoder_length]
      hparams: hyperparmeters for model.
      cache: dict, containing tensors which are the results of previous
          attentions, used for fast decoding.

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    decoder_output = transformer_decoder(
        decoder_input,
        encoder_output,
        raw_targets,
        decoder_self_attention_bias,
        encoder_decoder_attention_bias,
        hparams,
        cache=cache)

    # Expand since t2t expects 4d tensors.
    return tf.expand_dims(decoder_output, axis=2)

  def model_fn_body(self, features):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
          "tragets": Target decoder outputs.
              [batch_size, decoder_length, hidden_dim]
          "target_space_id"

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    inputs = features["inputs"]

    target_space = features["target_space_id"]
    encoder_output, encoder_decoder_attention_bias = self.encode(
        inputs, features["raw_inputs"], target_space, hparams)

    targets = features["targets"]
    targets = common_layers.flatten4d3d(targets)

    decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
        targets, hparams)

    return self.decode(
        decoder_input,
        encoder_output,
        features["raw_targets"],
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        hparams)


def transformer_prepare_encoder(inputs, target_space, hparams):
  """Prepare one shard of the model for the encoder.

  Args:
    inputs: a Tensor.
    target_space: a Tensor.
    hparams: run hyperparameters

  Returns:
    encoder_input: a Tensor, bottom of encoder stack
    encoder_self_attention_bias: a bias tensor for use in encoder self-attention
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
  """
  ishape_static = inputs.shape.as_list()
  encoder_input = inputs
  encoder_padding = common_attention.embedding_to_padding(encoder_input)
  ignore_padding = common_attention.attention_bias_ignore_padding(
      encoder_padding)
  encoder_self_attention_bias = ignore_padding
  encoder_decoder_attention_bias = ignore_padding
  if hparams.proximity_bias:
    encoder_self_attention_bias += common_attention.attention_bias_proximal(
        tf.shape(inputs)[1])
  # Append target_space_id embedding to inputs.
  emb_target_space = common_layers.embedding(
      target_space, 32, ishape_static[-1], name="target_space_embedding")
  emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
  encoder_input += emb_target_space
  return (encoder_input, encoder_self_attention_bias,
          encoder_decoder_attention_bias)


def transformer_prepare_decoder(targets, hparams):
  """Prepare one shard of the model for the decoder.

  Args:
    targets: a Tensor.
    hparams: run hyperparameters

  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a bias tensor for use in encoder self-attention
  """
  decoder_self_attention_bias = (
      common_attention.attention_bias_lower_triangle(tf.shape(targets)[1]))
  if hparams.proximity_bias:
    decoder_self_attention_bias += common_attention.attention_bias_proximal(
        tf.shape(targets)[1])
  decoder_input = common_layers.shift_right_3d(targets)
  return (decoder_input, decoder_self_attention_bias)


def embed_position_signal_sine(
    base_signal, channels, min_timescale=1.0, max_timescale=1.0e4):
  """Gets a bunch of sinusoids of different frequencies.

  This is a generalization of get_timing_signal_1d in common_attention
  which allows to use arbitrary base signals (not only range()).

  Args:
    signal: [batch_size, length] float32 tensor with the base signal. 
        Use [1, tf.range(length)] to recover get_timing_singal_1d.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float

  Returns:
    a Tensor of timing signals [batch_size1, length, channels]
  """
  num_timescales = channels // 2
  log_timescale_increment = (
      np.log(float(max_timescale) / float(min_timescale)) /
      (tf.to_float(num_timescales) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
  # inv_timescales 1/1000^0 1/1000^0.2 ...
  # base_signal [batch_size, length]
  scaled_time = tf.expand_dims(base_signal, -1) * tf.expand_dims(tf.expand_dims(inv_timescales, 0), 0)
  # scaled_time [batch_size, length, num_timescales]
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)
  signal = tf.pad(signal, [[0, 0], [0, 0], [0, tf.mod(channels, 2)]])
  return signal

def embed_position_signal_rnn(base_signal, channels, scope_name):
  with tf.variable_scope("pos_rnn_%s" % scope_name):
    cell = tf.contrib.rnn.BasicRNNCell(channels, activation=tf.tanh)
    max_length = tf.cast(tf.reduce_max(base_signal) + 1, tf.int32)
    inputs = tf.ones([1, max_length, 1])
    pos_embeds, _ = tf.nn.dynamic_rnn(
      cell,
      inputs,
      dtype=tf.float32,
      time_major=False)
    pos_embeds = tf.squeeze(pos_embeds, 0)
    # pos embeds has shape [max_pos, channels]
    embedded_signal = tf.gather(pos_embeds, tf.cast(base_signal, tf.int32))
    return embedded_signal    


BFS_SIGNAL_BFS = 0
BFS_SIGNAL_LOGBFS = 1
BFS_SIGNAL_NUM_LEFT_LOCAL = 2
BFS_SIGNAL_NUM_LEFT_GLOBAL = 3
BFS_SIGNAL_PARENT_TIMING = 4
BFS_SIGNAL_PARENT_BFS = 5
BFS_SIGNAL_PARENT_NUM_LEFT_GLOBAL = 6

def _generate_bfs_signals_py(raw_x, max_terminal_id, closing_bracket_id, max_children):
  # TODO: Implement this in tensorflow
  batch_size = raw_x.shape[0]
  length = raw_x.shape[1]
  signals = np.zeros([batch_size, length, 7], dtype=np.float32)
  log_max_children = np.log(max_children)
  for batch_idx in xrange(batch_size):
    depths = [[]]
    cur_depth = 0
    num_left_local = 0
    for pos in xrange(1, length):
      token = raw_x[batch_idx, pos]
      signals[batch_idx, pos, BFS_SIGNAL_NUM_LEFT_LOCAL] = num_left_local
      if cur_depth > 0:
        signals[batch_idx, pos, BFS_SIGNAL_PARENT_NUM_LEFT_GLOBAL] = len(depths[cur_depth-1])-1
        signals[batch_idx, pos, BFS_SIGNAL_PARENT_TIMING] = depths[cur_depth-1][-1]
        
      if token == closing_bracket_id:
        cur_depth -= 1
        num_left_local = 0
      elif token > max_terminal_id:
        depths[cur_depth].append(pos)
        cur_depth += 1
        num_left_local = 0
        if cur_depth == len(depths):
          depths.append([])
      else:
        depths[cur_depth].append(pos)
        num_left_local += 1
    bfs_number = 1
    for cur_depth, depth_positions in enumerate(depths):
      # We use log(x^d+y) = d log(x) + log(1+y/x^d)
      logbfs_incr = float(max_children) ** float(-cur_depth - 1)
      logbfs_offset = float(cur_depth + 1) * log_max_children
      for num_left_global, pos in enumerate(depth_positions):
        signals[batch_idx, pos, BFS_SIGNAL_BFS] = bfs_number
        signals[batch_idx, pos, BFS_SIGNAL_PARENT_BFS] = signals[batch_idx, int(signals[batch_idx, pos, BFS_SIGNAL_PARENT_TIMING]), BFS_SIGNAL_BFS] 
        signals[batch_idx, pos, BFS_SIGNAL_LOGBFS] = logbfs_offset + np.log(1.0 + num_left_global * logbfs_incr)
        signals[batch_idx, pos, BFS_SIGNAL_NUM_LEFT_GLOBAL] = num_left_global
        bfs_number += 1
  return signals


def generate_positional_signals(raw_x, hparams):
  """
  Args:
    raw_x: [batch, length]

  """
  shp = tf.shape(raw_x)
  signals = {}
  # explicitly broadcast timing signal (required for ffn)
  timing = tf.to_float(tf.range(shp[1]))
  signals["timing"] = tf.tile(tf.expand_dims(timing, 0), [shp[0], 1])
  try:
    is_closing = tf.equal(raw_x, hparams.closing_bracket_id)
    is_opening = tf.logical_and(
        tf.greater(raw_x, hparams.max_terminal_id),
        tf.logical_not(is_closing))
    is_opening_float = tf.cast(is_opening, tf.float32) 
    is_closing_float = tf.cast(is_closing, tf.float32)
    signals["depth"] = tf.cumsum(
        is_opening_float - is_closing_float, exclusive=True)
    signals["dfs"] = tf.cumsum(1.0 - is_closing_float)
    bfs_signals = tf.py_func(_generate_bfs_signals_py, 
        [raw_x, hparams.max_terminal_id, hparams.closing_bracket_id, hparams.max_children], tf.float32)
    bfs_signals.set_shape([None, None, 7])
    signals["bfs"] = bfs_signals[:, :, BFS_SIGNAL_BFS]
    signals["logbfs"] = bfs_signals[:, :, BFS_SIGNAL_LOGBFS]
    signals["num_left_local"] = bfs_signals[:, :, BFS_SIGNAL_NUM_LEFT_LOCAL]
    signals["num_left_global"] = bfs_signals[:, :, BFS_SIGNAL_NUM_LEFT_GLOBAL]
    signals["parent_timing"] = bfs_signals[:, :, BFS_SIGNAL_PARENT_TIMING]
    signals["parent_bfs"] = bfs_signals[:, :, BFS_SIGNAL_PARENT_BFS]
    signals["parent_num_left_global"] = bfs_signals[:, :, BFS_SIGNAL_PARENT_NUM_LEFT_GLOBAL]

    #for bla in ["bfs", "timing", "num_left_global", "parent_bfs", "parent_num_left_global", "parent_timing"]:
    #  tmp = tf.reduce_max(usr_utils.print_data(signals[bla][0,:], msg=bla, dtype=tf.float32))
    #  signals[bla] = signals[bla] + tmp - tmp
  except AttributeError:
    tf.logging.info("Do not generate tree based position signals "
                    "(max_terminal_id or closing_bracket_id not set)")
  return signals


def generate_positional_embeddings(pos_signals, pos_strategies, hparams):
  """Similar to common_attention.add_timing_signal_1d but with support for 
  advanced positional embedding strategies.

  Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float

  Returns:
    a Tensor the same shape as x.
  """
  if not pos_strategies or pos_strategies == "none":
    return None
  strategies = []
  max_timescale_sum = 0.0
  for strat in pos_strategies.split(","):
    try:
      strat, tmp = strat.split(":")
      max_timescale = float(tmp)
    except:
      max_timescale = 1.0e4
    if strat in pos_signals:
      strategies.append((strat, max_timescale))
      max_timescale_sum += max_timescale
    else:
      tf.logging.error("The position signal '%s' is unavailable and "
                       "will be ignored." % strat)
  embeddings = []
  #total_channels = tf.shape(x)[2]
  total_channels = hparams.hidden_size
  channels_sum = 0
  for strat, max_timescale in strategies:
    if hparams.multi_pos_policy == "sequential":
      channels = total_channels
    else: # parallel: separate channels for each signal
      channels = int(total_channels * max_timescale / max_timescale_sum)
      if channels <= 10.0:
          tf.logging.error("Number of channels for %s is critically low (%d)"
                           % (strat, channels))
      channels_sum += channels
    if hparams.pos_embed == "sine":
      embeddings.append(embed_position_signal_sine(
         pos_signals[strat], channels, max_timescale=max_timescale))
    elif hparams.pos_embed == "rnn":
      embeddings.append(embed_position_signal_rnn(
         pos_signals[strat], channels, strat))
    else:
      tf.logging.error("Unknown position embedding strategy '%s'" 
                       % hparams.pos_embed)
  if hparams.multi_pos_policy == "sequential":
    combined_embeddings = sum(embeddings)
  else: # parallel: concatenate embeddings, pad if necessary
    combined_embeddings = tf.concat(embeddings, axis=2)
    if total_channels != channels_sum:
      combined_embeddings = tf.pad(combined_embeddings,
          [[0, 0], [0, 0], [0, total_channels - channels_sum]])
  combined_embeddings.set_shape([None, None, total_channels])
  return combined_embeddings


def transformer_rnn_layer(x, sequence_length, hparams, bidirectional=False):
  if not bidirectional:
    cell = tf.contrib.rnn.BasicLSTMCell(hparams.hidden_size)
    y, _ = tf.nn.dynamic_rnn(
      cell,
      x,
      sequence_length=sequence_length,
      dtype=tf.float32,
      time_major=False)
  else:
    with tf.variable_scope("fwd"):
      fwd_cell = tf.contrib.rnn.BasicLSTMCell(hparams.hidden_size / 2)
    with tf.variable_scope("bwd"):
      bwd_cell = tf.contrib.rnn.BasicLSTMCell(hparams.hidden_size / 2)
    bi_y, _ = tf.nn.bidirectional_dynamic_rnn(
      fwd_cell, bwd_cell, x,
      sequence_length=sequence_length, 
      dtype=tf.float32,
      time_major=False)
    y = tf.concat(bi_y, -1)
  return y


def _iter_layer_types(layer_types_param, layer):
  for layer_type in layer_types_param.split(","):
    if ":" in layer_type:
      layer_type, max_layer = layer_type.split(":")
      if layer < int(max_layer):
        yield layer_type
    else:
      yield layer_type


def transformer_encoder(encoder_input,
                        raw_inputs,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder"):
  """A stack of transformer layers.

  Args:
    encoder_input: a Tensor
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string

  Returns:
    y: a Tensors
  """
  x = encoder_input
  with tf.variable_scope(name):
    raw_encoder_input = tf.squeeze(raw_inputs, axis=[-2, -1])
    sequence_length = usr_utils.get_length_from_raw(raw_encoder_input)  # Used for RNNs
    pos_signals = generate_positional_signals(raw_encoder_input, hparams)
    pos_embeddings = generate_positional_embeddings(pos_signals, hparams.encoder_pos, hparams)
    attention_pos_embeddings = generate_positional_embeddings(pos_signals, hparams.encoder_attention_pos, hparams)
    if "sum" in hparams.pos_integration:
      x = x + pos_embeddings
    elif "ffn" in hparams.pos_integration:
      with tf.variable_scope("pos_ffn"):
        x = tf.concat([x, pos_embeddings], axis=2)
        x = transformer_ffn_layer(x, hparams)
    pad_remover = None
    if hparams.use_pad_remover:
      pad_remover = expert_utils.PadRemover(
          common_attention.attention_bias_to_padding(
              encoder_self_attention_bias))
    for layer in xrange(hparams.num_encoder_layers or
                        hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        for layer_type in _iter_layer_types(hparams.encoder_layer_types, layer):
          if layer_type == "self_att":
            with tf.variable_scope("self_attention"):
              y = model_helper.multihead_attention_qkv(
                  common_layers.layer_preprocess(x, hparams),
                  None,
                  None,
                  encoder_self_attention_bias,
                  hparams.attention_key_channels or hparams.hidden_size,
                  hparams.attention_value_channels or hparams.hidden_size,
                  hparams.hidden_size,
                  hparams.num_heads,
                  hparams.attention_dropout,
                  attention_type=hparams.encoder_self_attention_type,
                  attention_order=hparams.attention_order,
                  max_relative_position=hparams.max_relative_position)
              x = common_layers.layer_postprocess(x, y, hparams)
          elif layer_type == "rnn":
            with tf.variable_scope("recurrent"):
              y = transformer_rnn_layer(
                  common_layers.layer_preprocess(x, hparams), 
                  sequence_length, 
                  hparams)
              x = common_layers.layer_postprocess(x, y, hparams)
          elif layer_type == "birnn":
            with tf.variable_scope("recurrent"):
              y = transformer_rnn_layer(
                  common_layers.layer_preprocess(x, hparams), 
                  sequence_length, 
                  hparams,
                  bidirectional=True)
              x = common_layers.layer_postprocess(x, y, hparams)
          elif layer_type == "pos_self_att" and attention_pos_embeddings is not None:
            with tf.variable_scope("pos_self_attention"):
              y = model_helper.multihead_attention_qkv(
                    attention_pos_embeddings,  # Query
                    attention_pos_embeddings,  # Key
                    common_layers.layer_preprocess(x, hparams), # Value
                    encoder_self_attention_bias,
                    hparams.attention_key_channels or hparams.hidden_size,
                    hparams.attention_value_channels or hparams.hidden_size,
                    hparams.hidden_size,
                    hparams.num_heads,
                    hparams.attention_dropout,
                    attention_type=hparams.pos_self_attention_type,
                    attention_order=hparams.attention_order,
                    max_relative_position=hparams.max_relative_position)
              x = common_layers.layer_postprocess(x, y, hparams)
          else:
            tf.logging.warn("Ignoring '%s' in encoder_layer_types" % layer_type)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams), hparams, pad_remover)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it shuold also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)


def transformer_decoder(decoder_input,
                        encoder_output,
                        raw_targets,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        name="decoder"):
  """A stack of transformer layers.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention
      (see common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
    name: a string

  Returns:
    y: a Tensors
  """
  x = decoder_input
  with tf.variable_scope(name):
    sequence_length = usr_utils.get_length_from_raw(
        tf.squeeze(raw_targets, axis=[-2, -1]))  # Used for RNNs
    sequence_length = sequence_length + 1  # Because of shifting 
    raw_decoder_input = common_layers.shift_right(raw_targets)
    raw_decoder_input = tf.squeeze(raw_decoder_input, axis=[-2, -1])
    pos_signals = generate_positional_signals(raw_decoder_input, hparams)
    pos_embeddings = generate_positional_embeddings(pos_signals, hparams.decoder_pos, hparams)
    attention_pos_embeddings = generate_positional_embeddings(pos_signals, hparams.decoder_attention_pos, hparams)
    if "sum" in hparams.pos_integration:
      x = x + pos_embeddings
    elif "ffn" in hparams.pos_integration:
      with tf.variable_scope("pos_ffn"):
        x = tf.concat([x, pos_embeddings], axis=2)
        x = transformer_ffn_layer(x, hparams)
    for layer in xrange(hparams.num_decoder_layers or
                        hparams.num_hidden_layers):
      layer_name = "layer_%d" % layer
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.variable_scope(layer_name):
        for layer_type in _iter_layer_types(hparams.decoder_layer_types, layer):
          if layer_type == "self_att":
            with tf.variable_scope("self_attention"):
              y = model_helper.multihead_attention_qkv(
                  common_layers.layer_preprocess(x, hparams),
                  None,
                  None,
                  decoder_self_attention_bias,
                  hparams.attention_key_channels or hparams.hidden_size,
                  hparams.attention_value_channels or hparams.hidden_size,
                  hparams.hidden_size,
                  hparams.num_heads,
                  hparams.attention_dropout,
                  attention_type=hparams.decoder_self_attention_type,
                  attention_order=hparams.attention_order,
                  max_relative_position=hparams.max_relative_position,
                  cache=layer_cache)
              x = common_layers.layer_postprocess(x, y, hparams)
          elif layer_type == "rnn":
            with tf.variable_scope("recurrent"):
              y = transformer_rnn_layer(
                  common_layers.layer_preprocess(x, hparams), 
                  sequence_length, 
                  hparams)
              x = common_layers.layer_postprocess(x, y, hparams)
          elif layer_type == "pos_self_att" and attention_pos_embeddings is not None:
            with tf.variable_scope("pos_self_attention"):
              y = model_helper.multihead_attention_qkv(
                  attention_pos_embeddings,  # Query
                  attention_pos_embeddings,  # Key
                  common_layers.layer_preprocess(x, hparams), # Value
                  decoder_self_attention_bias,
                  hparams.attention_key_channels or hparams.hidden_size,
                  hparams.attention_value_channels or hparams.hidden_size,
                  hparams.hidden_size,
                  hparams.num_heads,
                  hparams.attention_dropout,
                  attention_type=hparams.pos_self_attention_type,
                  attention_order=hparams.attention_order,
                  max_relative_position=hparams.max_relative_position)
            x = common_layers.layer_postprocess(x, y, hparams)
          elif layer_type == "enc_att" and encoder_output is not None:
            with tf.variable_scope("encdec_attention"):
              # TODO(llion): Add caching.
              y = model_helper.multihead_attention_qkv(
                  common_layers.layer_preprocess(x, hparams),
                  encoder_output,
                  None,
                  encoder_decoder_attention_bias,
                  hparams.attention_key_channels or hparams.hidden_size,
                  hparams.attention_value_channels or hparams.hidden_size,
                  hparams.hidden_size, hparams.num_heads,
                  hparams.attention_dropout)
              x = common_layers.layer_postprocess(x, y, hparams)
          else:
            tf.logging.warn("Ignoring '%s' in decoder_layer_types" % layer_type)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams), hparams)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it shuold also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)


def transformer_ffn_layer(x, hparams, pad_remover=None):
  """Feed-forward layer in the transformer.

  Args:
    x: a Tensor of shape [batch_size, length, hparams.hidden_size]
    hparams: hyperparmeters for model
    pad_remover: an expert_utils.PadRemover object tracking the padding
      positions. If provided, when using convolutional settings, the padding
      is removed before applying the convolution, and restored afterward. This
      can give a significant speedup.

  Returns:
    a Tensor of shape [batch_size, length, hparams.hidden_size]
  """
  if hparams.ffn_layer == "conv_hidden_relu":
    # In simple convolution mode, use `pad_remover` to speed up processing.
    if pad_remover:
      original_shape = tf.shape(x)
      # Collapse `x` across examples, and remove padding positions.
      x = tf.reshape(x, tf.concat([[-1], tf.shape(x)[2:]], axis=0))
      x = tf.expand_dims(pad_remover.remove(x), axis=0)
    conv_output = common_layers.conv_hidden_relu(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        dropout=hparams.relu_dropout)
    if pad_remover:
      # Restore `conv_output` to the original shape of `x`, including padding.
      conv_output = tf.reshape(
          pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
    return conv_output
  elif hparams.ffn_layer == "parameter_attention":
    return common_attention.parameter_attention(
        x, hparams.parameter_attention_key_channels or hparams.hidden_size,
        hparams.parameter_attention_value_channels or hparams.hidden_size,
        hparams.hidden_size, hparams.filter_size, hparams.num_heads,
        hparams.attention_dropout)
  elif hparams.ffn_layer == "conv_hidden_relu_with_sepconv":
    return common_layers.conv_hidden_relu(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        kernel_size=(3, 1),
        second_kernel_size=(31, 1),
        padding="LEFT",
        dropout=hparams.relu_dropout)
  else:
    assert hparams.ffn_layer == "none"
    return x

