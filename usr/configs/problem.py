# coding=utf-8
"""Additional problems for standard T2T models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators.translate_ende import TranslateEndeWmt32k
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_encoder
from usr import utils as usr_utils
import os

@registry.register_problem
class TranslateFlatBothtaggedEndeWmt32k(TranslateEndeWmt32k):
  @property
  def vocab_file(self):
    return "vocab.%s" % self.name

@registry.register_problem
class TranslateFlatStarttaggedEndeWmt32k(TranslateEndeWmt32k):
  @property
  def vocab_file(self):
    return "vocab.%s" % self.name


@registry.register_problem
class TranslateFlatStarttaggedPtb16k(TranslateEndeWmt32k):
  @property
  def vocab_file(self):
    return "vocab.%s" % self.name

  def hparams(self, defaults, model_hparams):
    super(TranslateFlatStarttaggedPtb16k, self).hparams(defaults, model_hparams)
    usr_utils.look_up_token_id(self._encoders["targets"], "closing_bracket_id", "##)##", model_hparams)
    usr_utils.extract_max_terminal_id(self._encoders["targets"], model_hparams)

    
@registry.register_problem
class TranslateFlatNtStarttaggedPtb16k(TranslateFlatStarttaggedPtb16k):

  @property
  def src_vocab_file(self):
    return "vocab.src.%s" % self.name

  @property
  def trg_vocab_file(self):
    return "vocab.trg.%s" % self.name

  def _create_encoder(self, data_dir, vocab_file):
    if self.is_character_level:
      encoder = text_encoder.ByteTextEncoder()
    elif self.use_subword_tokenizer:
      vocab_filename = os.path.join(data_dir, vocab_file)
      encoder = text_encoder.SubwordTextEncoder(vocab_filename)
    else:
      vocab_filename = os.path.join(data_dir, vocab_file)
      encoder = text_encoder.TokenTextEncoder(vocab_filename)
    return encoder
 
  def feature_encoders(self, data_dir):
    ret = {"targets": self._create_encoder(data_dir, self.trg_vocab_file)}
    if self.has_inputs:
      ret["inputs"] = self._create_encoder(data_dir, self.src_vocab_file)
    return ret

