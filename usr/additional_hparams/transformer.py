# coding=utf-8
"""Additional hparams for the transformer model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.utils import registry
from tensor2tensor.models.transformer import transformer_base


@registry.register_hparams
def transformer_base_12gb_gpu():
  """HParams for transformer base model for a single 12GB gpu."""
  hparams = transformer_base()
  hparams.learning_rate = 0.025
  hparams.learning_rate_warmup_steps = 16000
  hparams.batch_size = 8192
  return hparams

