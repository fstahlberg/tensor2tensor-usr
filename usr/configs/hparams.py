# coding=utf-8
"""Additional hparams for standard T2T models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import transformer_base_v2, transformer_big, transformer_relative_big
from tensor2tensor.models.slicenet import slicenet_params1_noam


@registry.register_hparams
def transformer_base_gradout():
  """Transformer with GradOut loss.
  
  Requires the T2T fork from https://github.com/fstahlberg/tensor2tensor
  """
  hparams = transformer_base_v2()
  hparams.target_modality = "symbol:gradout"
  return hparams


@registry.register_hparams
def transformer_base_gradout_large_batch():
  """Transformer with GradOut loss.
  
  Requires the T2T fork from https://github.com/fstahlberg/tensor2tensor
  """
  hparams = transformer_base_v2()
  hparams.target_modality = "symbol:gradout"
  hparams.optimizer_multistep_accumulate_steps = 8
  hparams.optimizer = "MultistepAdam"
  return hparams


@registry.register_hparams
def transformer_base_v2_fake_replicas():
  hparams = transformer_base_v2()
  hparams.learning_rate_warmup_steps *= 8
  hparams.learning_rate /= math.sqrt(8.0)
  hparams.max_length = 150
  hparams.batch_size = 8192
  return hparams


@registry.register_hparams
def transformer_base_v2_large_batch():
  hparams = transformer_base_v2()
  hparams.optimizer_multistep_accumulate_steps = 8
  hparams.optimizer = "MultistepAdam"
  return hparams


@registry.register_hparams
def transformer_base_v2_large_batch2():
  hparams = transformer_base_v2()
  hparams.optimizer_multistep_accumulate_steps = 2
  hparams.optimizer = "MultistepAdam"
  return hparams


@registry.register_hparams
def transformer_base_v2_large_batch4():
  hparams = transformer_base_v2()
  hparams.optimizer_multistep_accumulate_steps = 4
  hparams.optimizer = "MultistepAdam"
  return hparams

@registry.register_hparams
def transformer_base_v2_large_batch8():
  """Replication of Vaswani et al., 2017 on a single 12GB gpu."""
  hparams = transformer_base_v2()
  hparams.optimizer_multistep_accumulate_steps = 8
  hparams.optimizer = "MultistepAdam"
  return hparams

@registry.register_hparams
def transformer_big_large_batch4():
  hparams = transformer_big()
  hparams.batch_size = 2048
  hparams.optimizer_multistep_accumulate_steps = 4
  hparams.optimizer = "MultistepAdam"
  return hparams

@registry.register_hparams
def transformer_big_large_batch128():
  hparams = transformer_big()
  hparams.batch_size = 1024
  hparams.optimizer_multistep_accumulate_steps = 128
  hparams.optimizer = "MultistepAdam"
  return hparams

@registry.register_hparams
def transformer_relative_big_large_batch4():
  hparams = transformer_relative_big()
  hparams.batch_size = 2048
  hparams.optimizer_multistep_accumulate_steps = 4
  hparams.optimizer = "MultistepAdam"
  return hparams


@registry.register_hparams
def transformer_relative_big_large_batch128():
  hparams = transformer_relative_big()
  hparams.batch_size = 1024
  hparams.optimizer_multistep_accumulate_steps = 128
  hparams.optimizer = "MultistepAdam"
  return hparams


@registry.register_hparams
def transformer_base_v2_large_batch32():
  hparams = transformer_base_v2()
  hparams.optimizer_multistep_accumulate_steps = 32
  hparams.optimizer = "MultistepAdam"
  return hparams

@registry.register_hparams
def transformer_base_v2_small_lr():
  hparams = transformer_base_v2()
  hparams.learning_rate /= math.sqrt(8.0)
  hparams.max_length = 150
  hparams.batch_size = 8192
  return hparams

@registry.register_hparams
def slicenet_large_batch2():
  hparams = slicenet_params1_noam()
  hparams.optimizer_multistep_accumulate_steps = 2
  hparams.optimizer = "MultistepAdam"
  hparams.batch_size = 4096
  hparams.max_length = 150
  return hparams
