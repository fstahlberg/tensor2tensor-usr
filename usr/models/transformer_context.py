# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
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
"""Transformer model from "Attention Is All You Need".

The Transformer model consists of an encoder and a decoder. Both are stacks
of self-attention layers followed by feed-forward layers. This model yields
good results on a number of problems, especially in NLP and machine translation.

See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762) for the full
description of the model and the results obtained with its early version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import registry

from tensor2tensor.models.transformer import Transformer, features_to_nonpadding, transformer_ffn_layer, transformer_prepare_decoder

from usr import utils as usr_utils

import tensorflow as tf


@registry.register_model
class TransformerContext(Transformer):
  """Attention net.  See file docstring."""

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_biases,
             decoder_self_attention_biases,
             hparams,
             cache=None,
             decode_loop_step=None,
             nonpadding=None,
             losses=None):
    """Decode Transformer outputs from encoder representation.

    Args:
      decoder_input: inputs to bottom of the model.
          [batch_size, decoder_length, hidden_dim]
      encoder_output: Encoder representation.
          [batch_size, input_length, hidden_dim]
      encoder_decoder_attention_biases: Bias and mask weights for
          encoder-decoder attention. [batch_size, input_length]
      decoder_self_attention_biases: Bias and mask weights for decoder
          self-attention. [batch_size, decoder_length]
      hparams: hyperparameters for model.
      cache: dict, containing tensors which are the results of previous
          attentions, used for fast decoding.
      decode_loop_step: An integer, step number of the decoding loop.
          Only used for inference on TPU.
      nonpadding: optional Tensor with shape [batch_size, decoder_length]
      losses: optional list onto which to append extra training losses

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    decoder_output = transformer_decoder(
        decoder_input,
        encoder_output,
        decoder_self_attention_biases,
        encoder_decoder_attention_biases,
        hparams,
        cache=cache,
        decode_loop_step=decode_loop_step,
        nonpadding=nonpadding,
        save_weights_to=self.attention_weights,
        losses=losses)

    if (common_layers.is_xla_compiled() and
        hparams.mode == tf.estimator.ModeKeys.TRAIN):
      # TPU does not react kindly to extra dimensions.
      # TODO(noam): remove this once TPU is more forgiving of extra dims.
      return decoder_output
    else:
      # Expand since t2t expects 4d tensors.
      return tf.expand_dims(decoder_output, axis=2)

  def body(self, features):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs.
              [batch_size, input_length, 1, hidden_dim].
          "targets": Target decoder outputs.
              [batch_size, decoder_length, 1, hidden_dim]
          "target_space_id": A scalar int from data_generators.problem.SpaceID.

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    losses = []

    if self.has_input:
      raise AttributeError("Context transformer encoder not implemented")
      inputs = features["inputs"]
      target_space = features["target_space_id"]
      encoder_output, encoder_decoder_attention_biases = self.encode(
          inputs, target_space, hparams, features=features, losses=losses)
    else:
      encoder_output, encoder_decoder_attention_biases = (None, None)

    targets = features["targets"]
    targets_shape = common_layers.shape_list(targets)
    targets = common_layers.flatten4d3d(targets)
    decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
        targets, hparams, features=features)
    decoder_self_attention_biases = expand_bias_modes(
        decoder_self_attention_bias, features["targets_seg"])
    decoder_output = self.decode(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_biases,
        decoder_self_attention_biases,
        hparams,
        nonpadding=features_to_nonpadding(features, "targets"),
        losses=losses)

    expected_attentions = features.get("expected_attentions")
    if expected_attentions is not None:
      attention_loss = common_attention.encoder_decoder_attention_loss(
          expected_attentions, self.attention_weights,
          hparams.expected_attention_loss_type,
          hparams.expected_attention_loss_multiplier)
      return decoder_output, {"attention_loss": attention_loss}

    ret = tf.reshape(decoder_output, targets_shape)
    if losses:
      return ret, {"extra_loss": tf.add_n(losses)}
    else:
      return ret


def expand_bias_modes(bias, segmentation):
  shp = tf.shape(segmentation)
  batch_size, seq_len = shp[0], shp[1]
  intra_segment_bias = common_attention.attention_bias_same_segment(
        segmentation, segmentation)
  inter_segment_bias = tf.to_float(
      tf.greater(intra_segment_bias, -1.0)) * -1e9
  intra_segment_bias = tf.minimum(intra_segment_bias, bias)
  inter_segment_bias = tf.minimum(inter_segment_bias, bias)
  # Make sure that at least the first element is always not masked
  first_only = tf.reshape(tf.one_hot(0, seq_len, 0.0, -1e9), [1, 1,  1, seq_len])
  inter_segment_bias = tf.maximum(first_only, inter_segment_bias)
  return {"full": bias,
          "intra": intra_segment_bias,
          "inter": inter_segment_bias}


def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_biases,
                        encoder_decoder_attention_biases,
                        hparams,
                        cache=None,
                        decode_loop_step=None,
                        name="decoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True,
                        losses=None):
  """A stack of transformer layers.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_biases: bias Tensor for self-attention
      (see common_attention.attention_bias())
    encoder_decoder_attention_biases: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop.
        Only used for inference on TPU.
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used
      to mask out padding in convolutional layers.  We generally only
      need this mask for "packed" datasets, because for ordinary datasets,
      no padding is ever followed by nonpadding.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: optional list onto which to append extra training losses

  Returns:
    y: a Tensors
  """
  x = decoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))
  with tf.variable_scope(name):
    for layer in range(hparams.num_decoder_layers or hparams.num_hidden_layers):
      layer_name = "layer_%d" % layer
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.variable_scope(layer_name):
        for context_type in hparams.transformer_context_types:
          with tf.variable_scope("self_attention_%s" % context_type):
            y = common_attention.multihead_attention(
                common_layers.layer_preprocess(x, hparams),
                None,
                decoder_self_attention_biases[context_type],
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                attention_type=hparams.self_attention_type,
                max_relative_position=hparams.max_relative_position,
                heads_share_relative_embedding=(
                    hparams.heads_share_relative_embedding),
                add_relative_to_values=hparams.add_relative_to_values,
                save_weights_to=save_weights_to,
                cache=layer_cache,
                make_image_summary=make_image_summary,
                dropout_broadcast_dims=attention_dropout_broadcast_dims,
                max_length=hparams.get("max_length"),
                decode_loop_step=decode_loop_step,
                vars_3d=hparams.get("attention_variables_3d"))
            x = common_layers.layer_postprocess(x, y, hparams)
        if encoder_output is not None:
          with tf.variable_scope("encdec_attention"):
            y = common_attention.multihead_attention(
                common_layers.layer_preprocess(x, hparams),
                encoder_output,
                encoder_decoder_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                max_relative_position=hparams.max_relative_position,
                heads_share_relative_embedding=(
                    hparams.heads_share_relative_embedding),
                add_relative_to_values=hparams.add_relative_to_values,
                save_weights_to=save_weights_to,
                cache=layer_cache,
                make_image_summary=make_image_summary,
                dropout_broadcast_dims=attention_dropout_broadcast_dims,
                max_length=hparams.get("max_length"),
                vars_3d=hparams.get("attention_variables_3d"))
            x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams),
              hparams,
              conv_padding="LEFT",
              nonpadding_mask=nonpadding,
              losses=losses,
              cache=layer_cache,
              decode_loop_step=decode_loop_step)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)

