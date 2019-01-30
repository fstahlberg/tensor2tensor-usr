# coding=utf-8
"""Additional problems for standard T2T models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators.translate_ende import TranslateEndeWmt32k
from tensor2tensor.data_generators.lm1b import LanguagemodelLm1b32k
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_encoder
from usr import utils as usr_utils
import os

# START WMT19 ------------------------------------------------

# Base En-De model definition. Do not use directl
@registry.register_problem
class TranslateEndeWmt19(TranslateEndeWmt32k):

  @property
  def src_vocab_file(self):
    return "vocab.ende.wmt18"
  @property
  def trg_vocab_file(self):
    return "vocab.ende.wmt18"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

# No paracrawl, no backtranslation
@registry.register_problem
class TranslateEndeWmt19Base(TranslateEndeWmt19):
  pass



# END WMT19 ------------------------------------------------


@registry.register_problem
class TranslateNeenLmgec32k(TranslateEndeWmt32k):
  @property
  def vocab_file(self):
    return "vocab.en.32k"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(vocab_filename)}


@registry.register_problem
class LanguagemodelEnLmgec1b32k(LanguagemodelLm1b32k):
  @property
  def vocab_file(self):
    return "vocab.en.32k"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    return {"targets": text_encoder.TokenTextEncoder(vocab_filename)}

@registry.register_problem
class LanguagemodelEnLmgec1b30kWords(LanguagemodelLm1b32k):
  @property
  def vocab_file(self):
    return "vocab.en.30k.words"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    return {"targets": text_encoder.TokenTextEncoder(vocab_filename)}

@registry.register_problem
class LanguagemodelEnLmgec1bChar(LanguagemodelLm1b32k):
  @property
  def vocab_file(self):
    return "vocab.en.char"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    return {"targets": text_encoder.TokenTextEncoder(vocab_filename)}

# Start CF project
@registry.register_problem
class TranslateEsenScieloHealth(TranslateEndeWmt32k):
  @property
  def vocab_file(self):
    return "vocab.esen"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(vocab_filename)}

@registry.register_problem
class TranslateEsenScieloBio(TranslateEndeWmt32k):
  @property
  def vocab_file(self):
    return "vocab.esen"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(vocab_filename)}

@registry.register_problem
class TranslateEsenScieloHealthBio(TranslateEndeWmt32k):
  @property
  def vocab_file(self):
    return "vocab.esen"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(vocab_filename)}
# End CF project

@registry.register_problem
class TranslateDeenWmt32k(TranslateEndeWmt32k):
  pass

@registry.register_problem
class TranslateEnderWmt32k(TranslateEndeWmt32k):
  pass

@registry.register_problem
class TranslateFlatBothtaggedEndeWmt32k(TranslateEndeWmt32k):
  @property
  def vocab_file(self):
    return "vocab.%s" % self.name

  def hparams(self, defaults, model_hparams):
    super(TranslateFlatBothtaggedEndeWmt32k, self).hparams(defaults, model_hparams)
    usr_utils.extract_max_terminal_id(self._encoders["targets"], model_hparams)

@registry.register_problem
class TranslateFlatStarttaggedEndeWmt32k(TranslateEndeWmt32k):
  @property
  def vocab_file(self):
    return "vocab.%s" % self.name

  def hparams(self, defaults, model_hparams):
    super(TranslateFlatStarttaggedEndeWmt32k, self).hparams(defaults, model_hparams)
    usr_utils.look_up_token_id(self._encoders["targets"], "closing_bracket_id", "##)##", model_hparams)
    usr_utils.extract_max_terminal_id(self._encoders["targets"], model_hparams)


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


@registry.register_problem
class TranslatePtenScieloh(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.%s" % self.name

  @property
  def trg_vocab_file(self):
    return "vocab.%s" % self.name

  @property
  def vocab_file(self):
    return "vocab.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TranslatePtenScielohOsm(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.%s" % self.name

  @property
  def trg_vocab_file(self):
    return "vocab.%s" % self.name

  @property
  def vocab_file(self):
    return "vocab.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TranslatePtenScielohOsm2(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.%s" % self.name

  @property
  def trg_vocab_file(self):
    return "vocab.%s" % self.name

  @property
  def vocab_file(self):
    return "vocab.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TranslateEsenScieloh(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.%s" % self.name

  @property
  def trg_vocab_file(self):
    return "vocab.%s" % self.name

  @property
  def vocab_file(self):
    return "vocab.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TranslateEsenScielohOsm(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.%s" % self.name

  @property
  def trg_vocab_file(self):
    return "vocab.%s" % self.name

  @property
  def vocab_file(self):
    return "vocab.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TranslateEsenScielohOsm2(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.%s" % self.name

  @property
  def trg_vocab_file(self):
    return "vocab.%s" % self.name

  @property
  def vocab_file(self):
    return "vocab.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TranslateJaenKyoto32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateJaenWat32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateJaenWatOsm32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

  def hparams(self, defaults, model_hparams):
    super(TranslateJaenWatOsm32k, self).hparams(defaults, model_hparams)
    usr_utils.look_up_token_id(self._encoders["targets"], "pop_id", "<EOP>", model_hparams)


# Phrase-based OSM
@registry.register_problem
class TranslateJaenWatRpbosmbpe32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

  def hparams(self, defaults, model_hparams):
    super(TranslateJaenWatRpbosmbpe32k, self).hparams(defaults, model_hparams)
    usr_utils.look_up_token_id(self._encoders["targets"], "pop_id", "<SRC_POP1>", model_hparams)

@registry.register_problem
class TranslateJajaWatPbosmbpe32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.ja.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TranslateJaenWatPbosmbpe32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

  def hparams(self, defaults, model_hparams):
    super(TranslateJaenWatPbosmbpe32k, self).hparams(defaults, model_hparams)
    usr_utils.look_up_token_id(self._encoders["targets"], "pop_id", "<SRC_POP1>", model_hparams)


@registry.register_problem
class TranslateJaenWatOsmbpeNopop32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TranslateJaenWatNoosmbpe32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateJaenWatPushedPreordered32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateJaenWatPushed2Preordered32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TranslateJaenWatPreordered32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TranslateJaenWatPbpreordered32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateJaenWatPreorderedPop32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateJaenWatOsmbpePushedNolex32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.nolex"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateJaenWatOsmbpeNolex32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.nolex"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TranslateJaenWatPbosmbpeNolex32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.pbnolex"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateJaenWatOsmbpe32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

  def hparams(self, defaults, model_hparams):
    super(TranslateJaenWatOsmbpe32k, self).hparams(defaults, model_hparams)
    usr_utils.look_up_token_id(self._encoders["targets"], "pop_id", "<EOP>", model_hparams)

@registry.register_problem
class TranslateJaenWatNileOsmbpe32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

  def hparams(self, defaults, model_hparams):
    super(TranslateJaenWatNileOsmbpe32k, self).hparams(defaults, model_hparams)
    usr_utils.look_up_token_id(self._encoders["targets"], "pop_id", "<EOP>", model_hparams)

@registry.register_problem
class TranslateJaenWatOsmfert32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.fert.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TranslateJaenKyotoOsm32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

  def hparams(self, defaults, model_hparams):
    super(TranslateJaenKyotoOsm32k, self).hparams(defaults, model_hparams)
    usr_utils.look_up_token_id(self._encoders["targets"], "pop_id", "<EOP>", model_hparams)

@registry.register_problem
class TranslateJaenKyotoOsmbpe32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

  def hparams(self, defaults, model_hparams):
    super(TranslateJaenKyotoOsmbpe32k, self).hparams(defaults, model_hparams)
    usr_utils.look_up_token_id(self._encoders["targets"], "pop_id", "<EOP>", model_hparams)


@registry.register_problem
class TranslateFlatStarttaggedJaenWat32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.ja.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

  def hparams(self, defaults, model_hparams):
    super(TranslateFlatStarttaggedJaenWat32k, self).hparams(defaults, model_hparams)
    usr_utils.look_up_token_id(self._encoders["targets"], "closing_bracket_id", "##)##", model_hparams)
    usr_utils.extract_max_terminal_id(self._encoders["targets"], model_hparams)


@registry.register_problem
class TranslateEndeTcWmt32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.en.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.de.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateEndeTcbpeWmt32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateEnzhTcbpeWmt32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.en.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.zh.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateZhenTcbpeWmt32k(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.zh.%s" % self.name
  @property
  def trg_vocab_file(self):
    return "vocab.en.%s" % self.name

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

# WMT18
@registry.register_problem
class TranslateEndeOsmWmt18base(TranslateEndeWmt32k):

  @property
  def src_vocab_file(self):
    return "vocab.ende.wmt18.osm"
  @property
  def trg_vocab_file(self):
    return "vocab.ende.wmt18.osm"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateEndePbosmWmt18base(TranslateEndeWmt32k):

  @property
  def src_vocab_file(self):
    return "vocab.ende.wmt18.osm"
  @property
  def trg_vocab_file(self):
    return "vocab.ende.wmt18.osm"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TranslateEndeWmt18base(TranslateEndeWmt32k):

  @property
  def src_vocab_file(self):
    return "vocab.ende.wmt18"
  @property
  def trg_vocab_file(self):
    return "vocab.ende.wmt18"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TranslateEndeWmt18(TranslateEndeWmt32k):

  @property
  def src_vocab_file(self):
    return "vocab.ende.wmt18"
  @property
  def trg_vocab_file(self):
    return "vocab.ende.wmt18"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateDeenWmt18(TranslateEndeWmt32k):

  @property
  def src_vocab_file(self):
    return "vocab.ende.wmt18"
  @property
  def trg_vocab_file(self):
    return "vocab.ende.wmt18"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateEnzhWmt18(TranslateEndeWmt32k):

  @property
  def src_vocab_file(self):
    return "vocab.en.wmt18"
  @property
  def trg_vocab_file(self):
    return "vocab.zh.wmt18"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateZhenWmt18(TranslateEndeWmt32k):

  @property
  def src_vocab_file(self):
    return "vocab.zh.wmt18"
  @property
  def trg_vocab_file(self):
    return "vocab.en.wmt18"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateEnderWmt18(TranslateEndeWmt32k):

  @property
  def src_vocab_file(self):
    return "vocab.ende.wmt18"
  @property
  def trg_vocab_file(self):
    return "vocab.ende.wmt18"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateDeenrWmt18(TranslateEndeWmt32k):

  @property
  def src_vocab_file(self):
    return "vocab.ende.wmt18"
  @property
  def trg_vocab_file(self):
    return "vocab.ende.wmt18"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateEnzhrWmt18(TranslateEndeWmt32k):

  @property
  def src_vocab_file(self):
    return "vocab.en.wmt18"
  @property
  def trg_vocab_file(self):
    return "vocab.zh.wmt18"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


@registry.register_problem
class TranslateZhenrWmt18(TranslateEndeWmt32k):

  @property
  def src_vocab_file(self):
    return "vocab.zh.wmt18"
  @property
  def trg_vocab_file(self):
    return "vocab.en.wmt18"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

