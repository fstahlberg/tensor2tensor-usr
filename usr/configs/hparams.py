# coding=utf-8
"""Additional hparams for standard T2T models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import transformer_base_v2
from tensor2tensor.models.slicenet import slicenet_params1_noam


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
  """Replication of Vaswani et al., 2017 on a single 12GB gpu.
  
  Requires the T2T fork from https://github.com/fstahlberg/tensor2tensor
  """
  hparams = transformer_base_v2()
  #hparams.batch_size = 8192
  hparams.batch_size = 4096
  hparams.fake_gpu_multiplier = 8
  hparams.max_length = 150
  hparams.optimizer = "LargebatchAdam"
  return hparams


@registry.register_hparams
def transformer_base_v2_large_batch2():
  """Replication of Vaswani et al., 2017 on a single 12GB gpu.
  
  Requires the T2T fork from https://github.com/fstahlberg/tensor2tensor
  """
  hparams = transformer_base_v2()
  hparams.fake_gpu_multiplier = 2
  hparams.optimizer = "LargebatchAdam"
  return hparams


@registry.register_hparams
def transformer_base_v2_large_batch4():
  """Replication of Vaswani et al., 2017 on a single 12GB gpu.
  
  Requires the T2T fork from https://github.com/fstahlberg/tensor2tensor
  """
  hparams = transformer_base_v2()
  hparams.fake_gpu_multiplier = 4
  hparams.optimizer = "LargebatchAdam"
  return hparams

@registry.register_hparams
def transformer_base_v2_large_batch32():
  """Replication of Vaswani et al., 2017 on a single 12GB gpu.
  
  Requires the T2T fork from https://github.com/fstahlberg/tensor2tensor
  """
  hparams = transformer_base_v2()
  hparams.fake_gpu_multiplier = 32
  hparams.optimizer = "LargebatchAdam"
  return hparams

@registry.register_hparams
def slicenet_large_batch2():
  """Replication of Vaswani et al., 2017 on a single 12GB gpu.
  
  Requires the T2T fork from https://github.com/fstahlberg/tensor2tensor
  """
  hparams = slicenet_params1_noam()
  hparams.fake_gpu_multiplier = 2
  hparams.optimizer = "LargebatchAdam"
  #hparams.batch_size = 2048
  hparams.batch_size = 4096
  hparams.max_length = 150
  return hparams
