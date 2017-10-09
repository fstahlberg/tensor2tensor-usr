# coding=utf-8
"""Modality for target roots."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.layers import modalities
from tensor2tensor.utils import registry

@registry.register_symbol_modality("targetroots")
class TargetRootsSymbolModality(modalities.SymbolModality):
  @property
  def name(self):
    return "target_roots_modality_%d_%d" % (self._vocab_size, 
                                            self._body_input_depth)

