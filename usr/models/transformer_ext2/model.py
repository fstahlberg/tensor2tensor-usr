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
from tensor2tensor.data_generators import text_encoder

from usr.models.transformer_ext import model_helper
from usr import utils as usr_utils

import numpy as np

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.util import nest


@registry.register_model
class TransformerExt2(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  def encode(self, inputs, target_space, hparams, features=None):
    """Encode transformer inputs.

    Args:
      inputs: Transformer inputs [batch_size, input_length, hidden_dim]
      target_space: scalar, target space ID.
      hparams: hyperparmeters for model.
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encodre-decoder attention. [batch_size, input_length]
    """
    inputs = common_layers.flatten4d3d(inputs)

    encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
        transformer_prepare_encoder(
            inputs, target_space, hparams, features=features))

    encoder_input = tf.nn.dropout(encoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    encoder_output = transformer_encoder(
        encoder_input, self_attention_bias,
        hparams, nonpadding=_features_to_nonpadding(features, "inputs"))

    return encoder_output, encoder_decoder_attention_bias

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None,
             terminal_decoder_bias=None, 
             nonterminal_decoder_bias=None,
             nonpadding=None,
             pos_signals=None):
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
      nonpadding: optional Tensor with shape [batch_size, decoder_length]

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    decoder_output = transformer_decoder(
        decoder_input,
        encoder_output,
        decoder_self_attention_bias,
        encoder_decoder_attention_bias,
        hparams,
        cache=cache,
        terminal_decoder_bias=terminal_decoder_bias, 
        nonterminal_decoder_bias=nonterminal_decoder_bias,
        nonpadding=nonpadding,
        pos_signals=pos_signals)

    if hparams.use_tpu and hparams.mode == tf.estimator.ModeKeys.TRAIN:
      # TPU does not react kindly to extra dimensions.
      # TODO(noam): remove this once TPU is more forgiving of extra dims.
      return decoder_output
    else:
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

    inputs = features.get("inputs")
    encoder_output, encoder_decoder_attention_bias = (None, None)
    if inputs is not None:
      target_space = features["target_space_id"]
      encoder_output, encoder_decoder_attention_bias = self.encode(
          inputs, target_space, hparams, features=features)

    targets = features["targets"]
    targets = common_layers.flatten4d3d(targets)
    target_nonpadding = _features_to_nonpadding(features, "targets")
    if target_nonpadding is None:
      target_nonpadding = tf.squeeze(1.0 - tf.cast(tf.equal(features["targets_raw"], text_encoder.PAD_ID), tf.float32), [2, 3])
    try:
      if hparams.no_feedback:
        targets = tf.expand_dims(target_nonpadding, -1)
        tf.logging.info("Do not feed back targets!")
    except:
      pass
    decoder_input, decoder_self_attention_bias, terminal_decoder_bias, nonterminal_decoder_bias, pos_signals = transformer_prepare_decoder(
        targets, hparams, features=features)

    return self.decode(decoder_input, encoder_output,
                       encoder_decoder_attention_bias,
                       decoder_self_attention_bias, hparams,
                       nonpadding=target_nonpadding,
                       terminal_decoder_bias=terminal_decoder_bias, 
                       nonterminal_decoder_bias=nonterminal_decoder_bias,
                       pos_signals=pos_signals)


def transformer_prepare_encoder(inputs, target_space, hparams, features=None):
  """Prepare one shard of the model for the encoder.

  Args:
    inputs: a Tensor.
    target_space: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well.
      This is needed now for "packed" datasets.

  Returns:
    encoder_input: a Tensor, bottom of encoder stack
    encoder_self_attention_bias: a bias tensor for use in encoder self-attention
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
  """
  ishape_static = inputs.shape.as_list()
  encoder_input = inputs
  if features and "inputs_segmentation" in features:
    # Packed dataset.  Keep the examples from seeing each other.
    inputs_segmentation = features["inputs_segmentation"]
    inputs_position = features["inputs_position"]
    targets_segmentation = features["targets_segmentation"]
    encoder_self_attention_bias = common_attention.attention_bias_same_segment(
        inputs_segmentation, inputs_segmentation)
    encoder_decoder_attention_bias = (
        common_attention.attention_bias_same_segment(
            targets_segmentation, inputs_segmentation))
  else:
    # Usual case - not a packed dataset.
    encoder_padding = common_attention.embedding_to_padding(encoder_input)
    ignore_padding = common_attention.attention_bias_ignore_padding(
        encoder_padding)
    encoder_self_attention_bias = ignore_padding
    encoder_decoder_attention_bias = ignore_padding
    inputs_position = None
  if hparams.proximity_bias:
    encoder_self_attention_bias += common_attention.attention_bias_proximal(
        common_layers.shape_list(inputs)[1])
  # Append target_space_id embedding to inputs.
  emb_target_space = common_layers.embedding(
      target_space, 32, ishape_static[-1], name="target_space_embedding")
  emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
  encoder_input += emb_target_space
  #if hparams.pos == "timing":
  #  if inputs_position is not None:
  #    encoder_input = common_attention.add_timing_signal_1d_given_position(
  #        encoder_input, inputs_position)
  #  else:
  #    encoder_input = common_attention.add_timing_signal_1d(encoder_input)
  raw_encoder_input = tf.squeeze(features['inputs_raw'], axis=[-2, -1])
  pos_signals = generate_positional_signals(raw_encoder_input, hparams)
  pos_embeddings = generate_positional_embeddings(pos_signals, hparams.encoder_pos, hparams)
  if "sum" in hparams.encoder_pos_integration:
    encoder_input = encoder_input + pos_embeddings
  elif "ffn" in hparams.encoder_pos_integration:
    with tf.variable_scope("encoder_pos_ffn"):
      encoder_input = tf.concat([encoder_input, pos_embeddings], axis=2)
      encoder_input = transformer_ffn_layer(encoder_input, hparams, conv_padding="SAME")
  return (encoder_input, encoder_self_attention_bias,
          encoder_decoder_attention_bias)


def transformer_prepare_decoder(targets, hparams, features=None):
  """Prepare one shard of the model for the decoder.

  Args:
    targets: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well.
      This is needed now for "packed" datasets.

  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a bias tensor for use in encoder self-attention
  """
  decoder_self_attention_bias = (
      common_attention.attention_bias_lower_triangle(
          common_layers.shape_list(targets)[1]))
  if features and "targets_segmentation" in features:
    # "Packed" dataset - keep the examples from seeing each other.
    targets_segmentation = features["targets_segmentation"]
    targets_position = features["targets_position"]
    decoder_self_attention_bias += common_attention.attention_bias_same_segment(
        targets_segmentation, targets_segmentation)
  else:
    targets_position = None
  if hparams.proximity_bias:
    decoder_self_attention_bias += common_attention.attention_bias_proximal(
        common_layers.shape_list(targets)[1])
  decoder_input = common_layers.shift_right_3d(targets)
  #if hparams.pos == "timing":
  #  if targets_position is not None:
  #    decoder_input = common_attention.add_timing_signal_1d_given_position(
  #        decoder_input, targets_position)
  #  else:
  #    decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  raw_decoder_input = common_layers.shift_right(features['targets_raw'])
  terminal_decoder_bias, nonterminal_decoder_bias = _get_t_nt_bias(
      raw_decoder_input, hparams, decoder_self_attention_bias)
  raw_decoder_input = tf.squeeze(raw_decoder_input, axis=[-2, -1])
  pos_signals = generate_positional_signals(
      raw_decoder_input, hparams, terminal_decoder_bias, nonterminal_decoder_bias)
  pos_embeddings = generate_positional_embeddings(pos_signals, hparams.decoder_pos, hparams)
  if "sum" in hparams.decoder_pos_integration:
    decoder_input = decoder_input + pos_embeddings
  elif "ffn" in hparams.decoder_pos_integration:
    with tf.variable_scope("decoder_pos_ffn"):
      decoder_input = tf.concat([decoder_input, pos_embeddings], axis=2)
      decoder_input = transformer_ffn_layer(decoder_input, hparams, conv_padding="LEFT")
  return (decoder_input, decoder_self_attention_bias, terminal_decoder_bias, nonterminal_decoder_bias, pos_signals)


def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder",
                        nonpadding=None):
  """A stack of transformer layers.

  Args:
    encoder_input: a Tensor
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This must either be
      passed in, which we do for "packed" datasets, or inferred from
      encoder_self_attention_bias.  The knowledge about padding is used
      for pad_remover(efficiency) and to mask out padding in convoltutional
      layers.

  Returns:
    y: a Tensors
  """
  x = encoder_input
  with tf.variable_scope(name):
    if nonpadding is not None:
      padding = 1.0 - nonpadding
    else:
      padding = common_attention.attention_bias_to_padding(
          encoder_self_attention_bias)
      nonpadding = 1.0 - padding
    pad_remover = None
    if hparams.use_pad_remover:
      pad_remover = expert_utils.PadRemover(padding)
    sequence_length = usr_utils.get_length_from_nonpadding(nonpadding)
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
          else:
            tf.logging.warn("Ignoring '%s' in encoder_layer_types" % layer_type)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams), hparams, pad_remover,
              conv_padding="SAME", nonpadding_mask=nonpadding)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it shuold also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)


def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        name="decoder",
                        terminal_decoder_bias=None, 
                        nonterminal_decoder_bias=None,
                        nonpadding=None,
                        pos_signals=None):
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
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used
      to mask out padding in convoltutional layers.  We generally only
      need this mask for "packed" datasets, because for ordinary datasets,
      no padding is ever followed by nonpadding.

  Returns:
    y: a Tensors
  """
  x = decoder_input
  sequence_length = usr_utils.get_length_from_nonpadding(nonpadding)
  with tf.variable_scope(name):
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
          elif layer_type == "nt_self_att":
            with tf.variable_scope("nonterminal_self_attention"):
              y = model_helper.multihead_attention_qkv(
                  common_layers.layer_preprocess(x, hparams),
                  None,
                  None,
                  nonterminal_decoder_bias,
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
          elif layer_type == "t_self_att":
            with tf.variable_scope("terminal_self_attention"):
              y = model_helper.multihead_attention_qkv(
                  common_layers.layer_preprocess(x, hparams),
                  None,
                  None,
                  terminal_decoder_bias,
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
          elif layer_type == "parent_ffn":
            with tf.variable_scope("parent_ffn"):
              parent_pointers = tf.cast(pos_signals["parent_timing"], tf.int32)
              parent_x = usr_utils.gather_2d(x, parent_pointers)
              x = tf.concat([x, parent_x], axis=2)
              x = transformer_ffn_layer(
                  x, hparams,
                  conv_padding="LEFT")
          elif layer_type == "rnn":
            with tf.variable_scope("recurrent"):
              y = transformer_rnn_layer(
                  common_layers.layer_preprocess(x, hparams), 
                  sequence_length, 
                  hparams)
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
              common_layers.layer_preprocess(x, hparams), hparams,
              conv_padding="LEFT", nonpadding_mask=nonpadding)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it shuold also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)


def transformer_ffn_layer(x,
                          hparams,
                          pad_remover=None,
                          conv_padding="LEFT",
                          nonpadding_mask=None):
  """Feed-forward layer in the transformer.

  Args:
    x: a Tensor of shape [batch_size, length, hparams.hidden_size]
    hparams: hyperparmeters for model
    pad_remover: an expert_utils.PadRemover object tracking the padding
      positions. If provided, when using convolutional settings, the padding
      is removed before applying the convolution, and restored afterward. This
      can give a significant speedup.
    conv_padding: a string - either "LEFT" or "SAME".
    nonpadding_mask: an optional Tensor with shape [batch_size, length].
      needed for convolutoinal layers with "SAME" padding.
      Contains 1.0 in positions corresponding to nonpadding.

  Returns:
    a Tensor of shape [batch_size, length, hparams.hidden_size]
  """
  ffn_layer = hparams.ffn_layer
  if ffn_layer == "conv_hidden_relu":
    # Backwards compatibility
    ffn_layer = "dense_relu_dense"
  if ffn_layer == "dense_relu_dense":
    # In simple convolution mode, use `pad_remover` to speed up processing.
    if pad_remover:
      original_shape = common_layers.shape_list(x)
      # Collapse `x` across examples, and remove padding positions.
      x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
      x = tf.expand_dims(pad_remover.remove(x), axis=0)
    conv_output = common_layers.dense_relu_dense(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        dropout=hparams.relu_dropout)
    if pad_remover:
      # Restore `conv_output` to the original shape of `x`, including padding.
      conv_output = tf.reshape(
          pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
    return conv_output
  elif ffn_layer == "conv_relu_conv":
    return common_layers.conv_relu_conv(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        first_kernel_size=3,
        second_kernel_size=1,
        padding=conv_padding,
        nonpadding_mask=nonpadding_mask,
        dropout=hparams.relu_dropout)
  elif ffn_layer == "parameter_attention":
    return common_attention.parameter_attention(
        x, hparams.parameter_attention_key_channels or hparams.hidden_size,
        hparams.parameter_attention_value_channels or hparams.hidden_size,
        hparams.hidden_size, hparams.filter_size, hparams.num_heads,
        hparams.attention_dropout)
  elif ffn_layer == "conv_hidden_relu_with_sepconv":
    return common_layers.conv_hidden_relu(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        kernel_size=(3, 1),
        second_kernel_size=(31, 1),
        padding="LEFT",
        dropout=hparams.relu_dropout)
  else:
    assert ffn_layer == "none"
    return x

def _features_to_nonpadding(features, inputs_or_targets="inputs"):
  key = inputs_or_targets + "_segmentation"
  if features and key in features:
    return tf.minimum(features[key], 1.0)
  return None

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
  with tf.variable_scope("pos_rnn_%s" % scope_name, reuse=tf.AUTO_REUSE):
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


def generate_timing_from_bias(bias):
  return tf.reduce_sum(tf.cast(tf.equal(tf.squeeze(bias, 1), 0.0), tf.float32), axis=-1) - 1.0

def generate_positional_signals(raw_x, hparams, terminal_mask=None, nonterminal_mask=None):
  """
  Args:
    raw_x: [batch, length]
    hparams: Hyper parameters
    terminal_mask: A [batch_size, 1, max_seq_len, max_seq_len] float32 tensor 
      or None. See _get_t_nt_mask()
    nonterminal_mask: A [batch_size, 1, max_seq_len, max_seq_len] float32 tensor
      or None. See _get_t_nt_mask()
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
        is_opening_float - is_closing_float, axis=1, exclusive=True)
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
    if terminal_mask is not None:
      signals["t_timing"] = generate_timing_from_bias(terminal_mask)
    if nonterminal_mask is not None:
      signals["nt_timing"] = generate_timing_from_bias(nonterminal_mask)
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


def _get_t_nt_bias(labels, hparams, mask=None):
  """Creates attention bias variables for terminals and nonterminals.

  Args:
    labels: A [batch_size, max_seq_len, 1, 1] int32 tensor with the
      raw target labels.
    hparams: hyper parameters.
    mask: A [batch_size, 1, max_seq_len, max_seq_len] float32 tensor
      with a prior mask, or None. Supports broadcasting. By convention,
      the mask uses large negative values to exclude entries, and zero
      otherwise

  Returns:
    Two [batch_size, 1, max_seq_len, max_seq_len] float32 tensors, one
    for terminal labels and one for nonterminal labels, ie. an entry is
    0.0 iff. mask is 0.0 and the corresponding label is a [non]terminal
    Otherwise, entries are set to -1e9. Returns `mask` if hparams do 
    not contain max_terminal_id.
  """
  try:
    labels = tf.transpose(labels, perm=[0, 2, 3, 1])
    t_mask = tf.less_equal(labels, hparams.max_terminal_id)
    is_pad = tf.equal(labels, text_encoder.PAD_ID)
    nt_mask = tf.logical_or(tf.logical_not(t_mask), is_pad)
    nt_mask = -1e9 * (1.0 - tf.cast(nt_mask, tf.float32))
    t_mask = -1e9 * (1.0 - tf.cast(t_mask, tf.float32))
    if mask is not None:
      nt_mask = tf.minimum(nt_mask, mask)
      t_mask = tf.minimum(t_mask, mask)
    return t_mask, nt_mask
  except AttributeError:
    tf.logging.info("t_self_att and nt_self_att back off to self_att "
                    "(max_terminal_id not set)")
    return mask, mask

