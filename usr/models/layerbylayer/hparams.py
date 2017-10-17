# coding=utf-8
"""Hyper parameter sets for layer-by-layer models."""

from tensor2tensor.utils import registry
from usr.models.tfnmt import hparams_helper
from tensor2tensor.layers import common_hparams


@registry.register_hparams
def transformer_layerbylayer_default():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.norm_type = "layer"
  hparams.hidden_size = 512
  #hparams.batch_size = 4096
  hparams.batch_size = 8192
  hparams.max_length = 256
  hparams.dropout = 0.0
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
  hparams.shared_embedding_and_softmax_weights = int(True)

  # Add new ones like this.
  hparams.add_hparam("filter_size", 2048)
  # Layer-related flags. If zero, these fall back on hparams.num_hidden_layers.
  hparams.add_hparam("num_encoder_layers", 0)
  hparams.add_hparam("num_decoder_layers", 0)
  # Attention-related flags.
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  hparams.add_hparam("ffn_layer", "conv_hidden_relu")
  hparams.add_hparam("parameter_attention_key_channels", 0)
  hparams.add_hparam("parameter_attention_value_channels", 0)
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("nbr_decoder_problems", 1)
  hparams.add_hparam("proximity_bias", int(False))
  hparams.add_hparam("use_pad_remover", int(True))
  hparams.add_hparam("self_attention_type", "dot_product")
  hparams.add_hparam("max_relative_position", 0)

  # Layerbylayer defaults
  hparams.add_hparam("target_root_attention", "pop")
  return hparams


@registry.register_hparams
def tfnmt_layerbylayer_default():
  hparams = hparams_helper.tfnmt_default()
  hparams.learning_rate_warmup_steps = 6000
  hparams.label_smoothing = 0.1
  hparams.batch_size = 8192
  hparams.max_input_seq_length = 100
  hparams.max_target_seq_length = 200  # Account for POP
  hparams.add_hparam("target_root_attention", "pop")
  hparams.add_hparam("target_root_encoder_type", "id")
  return hparams


@registry.register_hparams
def tfnmt_layerbylayer_bi():
  hparams = tfnmt_layerbylayer_default()
  hparams.max_target_seq_length = 100
  hparams.target_root_attention = "luong"
  hparams.target_root_encoder_type = "bi"
  return hparams


@registry.register_hparams
def tfnmt_layerbylayer_bipop():
  hparams = tfnmt_layerbylayer_default()
  hparams.target_root_encoder_type = "bi"
  return hparams


