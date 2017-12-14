# coding=utf-8
"""Additional hparams for standard T2T models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import transformer_base_v2


@registry.register_hparams
def transformer_base_v2_fake_replicas():
  hparams = transformer_base_v2()
  hparams.learning_rate_warmup_steps *= 8
  hparams.learning_rate /= math.sqrt(8.0)
  hparams.max_length = 150
  hparams.batch_size = 8192
  return hparams


@registry.register_hparams
def transformer_base_12gb_gpu_large_batch():
  """Replication of Vaswani et al., 2017 on a single 12GB gpu.
  
  Requires the T2T fork from https://github.com/fstahlberg/tensor2tensor
  """
  hparams = transformer_base()
  hparams.batch_size = 8192
  hparams.fake_gpu_multiplier = 4
  hparams.optimizer = "LargebatchAdam"
  return hparams


