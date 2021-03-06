# coding=utf-8

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

import tensorflow as tf


@registry.register_hparams
def transformer_ext2_v1():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.norm_type = "layer"
  hparams.hidden_size = 512
  hparams.batch_size = 4096
  hparams.max_length = 256
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer_gain = 1.0
  hparams.num_hidden_layers = 6
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.num_sampled_classes = 0
  hparams.label_smoothing = 0.1
  hparams.shared_embedding_and_softmax_weights = True
  # Add new ones like this.
  hparams.add_hparam("filter_size", 2048)
  # Layer-related flags. If zero, these fall back on hparams.num_hidden_layers.
  hparams.add_hparam("num_encoder_layers", 0)
  hparams.add_hparam("num_decoder_layers", 0)
  # Attention-related flags.
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  hparams.add_hparam("ffn_layer", "dense_relu_dense")
  hparams.add_hparam("parameter_attention_key_channels", 0)
  hparams.add_hparam("parameter_attention_value_channels", 0)
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("nbr_decoder_problems", 1)
  hparams.add_hparam("proximity_bias", False)
  hparams.add_hparam("use_pad_remover", True)
  hparams.add_hparam("encoder_self_attention_type", "dot_product")
  hparams.add_hparam("decoder_self_attention_type", "dot_product")
  hparams.add_hparam("max_relative_position", 0)

  # Advanced positional embeddings
  # Comma separated: timing, t_timing, nt_timing, bfs, logbfs, depth, num_left_local, num_left_global
  # Use eg "timing:1000" to specify max_timescale explicitly
  hparams.add_hparam("pos_embed", "sine") # sine, rnn
  hparams.add_hparam("multi_pos_policy", "parallel") # parallel, sequential
  hparams.add_hparam("max_children", 5) # for logbfs
  hparams.add_hparam("encoder_pos_integration", "sum") # sum, ffn
  hparams.add_hparam("decoder_pos_integration", "sum") # sum, ffn
  hparams.add_hparam("encoder_pos", "timing")  
  hparams.add_hparam("decoder_pos", "timing")  
  hparams.add_hparam("encoder_attention_pos", "")  
  hparams.add_hparam("decoder_attention_pos", "")
  hparams.add_hparam("pos_self_attention_type", "dot_product")
  # self_att, rnn, birnn, enc_att, t_self_att, nt_self_att, parent_ffn
  # eg. rnn:2 inserts a RNN layer in the two top layers
  hparams.add_hparam("encoder_layer_types", "self_att")
  hparams.add_hparam("decoder_layer_types", "self_att,enc_att")
  # Used for higher order attention types
  hparams.add_hparam("attention_order", 2)
  return hparams


@registry.register_hparams
def transformer_ext2_v2():
  hparams = transformer_ext2_v1()
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  hparams.learning_rate_warmup_steps = 8000
  hparams.learning_rate = 0.2
  return hparams


@registry.register_hparams
def transformer_ext2():
  return transformer_ext2_v2()


