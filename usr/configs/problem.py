# coding=utf-8
"""Additional problems for standard T2T models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators.wmt import TranslateEndeWmt32k
from tensor2tensor.utils import registry

@registry.register_problem
class TranslateFlatBothtaggedEndeWmt32k(TranslateEndeWmt32k):
  @property
  def vocab_file(self):
    return "vocab.%s" % self.name

