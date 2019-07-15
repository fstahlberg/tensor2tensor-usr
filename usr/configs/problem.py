# coding=utf-8
"""Additional problems for standard T2T models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators.translate_ende import TranslateEndeWmt32k
from tensor2tensor.data_generators.lm1b import LanguagemodelLm1b32k
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators.problem import preprocess_example_common
from usr import utils as usr_utils
import os

import tensorflow as tf
import numpy as np
import random

# START WMT19 ------------------------------------------------

# LM

# Base LM problem definition. Do not use directly
@registry.register_problem
class LanguagemodelEnWmt19(LanguagemodelLm1b32k):
  @property
  def vocab_file(self):
    return "vocab.ende.wmt19"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    return {"targets": text_encoder.TokenTextEncoder(vocab_filename)}


# news2016-2018
@registry.register_problem
class LanguagemodelDeWmt19News1618(LanguagemodelEnWmt19):
  pass

@registry.register_problem
class LanguagemodelEnWmt19News1618(LanguagemodelEnWmt19):
  pass


# Document-level news2016-2018 (glued with <s>)
@registry.register_problem
class LanguagemodelGlueDeWmt19News1618(LanguagemodelEnWmt19):
  pass


# Document-level news2016-2018 (glued with <s>, with targets_segmentation)
@registry.register_problem
class LanguagemodelGlueSegDeWmt19News1618(LanguagemodelEnWmt19):
  def example_reading_spec(self):
    data_fields = {
      "targets": tf.VarLenFeature(tf.int64),
      "targets_pos": tf.VarLenFeature(tf.int64),
      "targets_seg": tf.VarLenFeature(tf.int64),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)


# Document-level news2016-2018 (glued with <s>, with targets_segmentation)
@registry.register_problem
class LanguagemodelGlueSegEnWmt19News1618(LanguagemodelEnWmt19):
  def example_reading_spec(self):
    data_fields = {
      "targets": tf.VarLenFeature(tf.int64),
      "targets_pos": tf.VarLenFeature(tf.int64),
      "targets_seg": tf.VarLenFeature(tf.int64),
    }
    data_items_to_decoders = None
    return (data_fields, data_items_to_decoders)


# TM

# Base En-De problem definition. Do not use directly
@registry.register_problem
class TranslateEndeWmt19(TranslateEndeWmt32k):

  @property
  def src_vocab_file(self):
    return "vocab.ende.wmt19"
  @property
  def trg_vocab_file(self):
    return "vocab.ende.wmt19"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}


# No paracrawl, no backtranslation
@registry.register_problem
class TranslateEndeWmt19Base(TranslateEndeWmt19):
  pass


@registry.register_problem
class TranslateDeenWmt19Base(TranslateEndeWmt19):
  pass


# Full paracrawl, 4x oversampling, no backtranslation
@registry.register_problem
class TranslateEndeWmt19FullpcOs4(TranslateEndeWmt19):
  pass


# Naively (WMT18) filtered paracrawl, no oversampling, no backtranslation
@registry.register_problem
class TranslateEndeWmt19Naivepc(TranslateEndeWmt19):
  pass


# Naively (WMT18) filtered paracrawl, 2x oversampling, no backtranslation
@registry.register_problem
class TranslateEndeWmt19NaivepcOs2(TranslateEndeWmt19):
  pass


# ms-filtered paracrawl 5M, no backtranslation
@registry.register_problem
class TranslateEndeWmt19Mspc5(TranslateEndeWmt19):
  pass

# ms-filtered paracrawl 8M, no backtranslation
@registry.register_problem
class TranslateEndeWmt19Mspc8(TranslateEndeWmt19):
  pass

# Naively (WMT18) ms-filtered paracrawl 15M, no backtranslation
@registry.register_problem
class TranslateEndeWmt19Msnaivepc15(TranslateEndeWmt19):
  pass

# Naively (WMT18) ms-filtered paracrawl 10M, no backtranslation
@registry.register_problem
class TranslateEndeWmt19Msnaivepc10(TranslateEndeWmt19):
  pass


# ms-filtered paracrawl 10M, 2x oversampling, no backtranslation
@registry.register_problem
class TranslateEndeWmt19Mspc10Os2(TranslateEndeWmt19):
  pass

# ms-filtered paracrawl 15M, 2x oversampling, no backtranslation
@registry.register_problem
class TranslateEndeWmt19Mspc15Os2(TranslateEndeWmt19):
  pass


# Naively (WMT18) filtered paracrawl, no oversampling, news2017 backtranslation
@registry.register_problem
class TranslateEndeWmt19NaivepcBt17(TranslateEndeWmt19):
  pass


# Naively (WMT18) filtered paracrawl, 2x oversampling, news2017 backtranslation
@registry.register_problem
class TranslateEndeWmt19NaivepcOs2Bt17(TranslateEndeWmt19):
  pass

# Naively (WMT18) filtered paracrawl, 2x oversampling, news2018 backtranslation
@registry.register_problem
class TranslateEndeWmt19NaivepcOs2Bt18(TranslateEndeWmt19):
  pass

# Naively (WMT18) 2x filtered paracrawl, 3x rest, news2018 backtranslation
@registry.register_problem
class TranslateEndeWmt19NaivepcOs32Bt18(TranslateEndeWmt19):
  pass

# Naively (WMT18) filtered paracrawl, 2x oversampling, news2017-2018 backtranslation
@registry.register_problem
class TranslateEndeWmt19NaivepcOs2Bt1718(TranslateEndeWmt19):
  pass


# Naively (WMT18) filtered paracrawl, 2x oversampling, news2017 FBNoise backtranslation
@registry.register_problem
class TranslateEndeWmt19NaivepcOs2Fbnoisebt17(TranslateEndeWmt19):
  pass

# Naively (WMT18) filtered paracrawl, 2x oversampling, news2018 FBNoise backtranslation
@registry.register_problem
class TranslateEndeWmt19NaivepcOs2Fbnoisebt18(TranslateEndeWmt19):
  pass

# Naively (WMT18) filtered paracrawl, 2x oversampling, news2017+2018 FBNoise backtranslation
@registry.register_problem
class TranslateEndeWmt19NaivepcOs2Fbnoisebt1718(TranslateEndeWmt19):
  pass

# Naively (WMT18) 2x filtered paracrawl, 3x oversampling rest, news2017+2018 FBNoise backtranslation
@registry.register_problem
class TranslateEndeWmt19NaivepcOs32Fbnoisebt1718(TranslateEndeWmt19):
  pass

# Naively (WMT18) 2x filtered paracrawl, 3x oversampling rest, news2016+2017 FBNoise backtranslation
@registry.register_problem
class TranslateEndeWmt19NaivepcOs32Fbnoisebt1617(TranslateEndeWmt19):
  pass

# Naively (WMT18) 2x filtered paracrawl, 3x oversampling rest, news2016+2017+2018 FBNoise backtranslation
@registry.register_problem
class TranslateEndeWmt19NaivepcOs32Fbnoisebt1618(TranslateEndeWmt19):
  pass




# Naively (WMT18) 2x filtered paracrawl, 3x oversampling rest, news2016+2017+2018 FBNoise filtered backtranslation
@registry.register_problem
class TranslateEndeWmt19NaivepcOs32Fbnoisebtfilt1618(TranslateEndeWmt19):
  pass

# Naively (WMT18) 2x filtered paracrawl, 3x oversampling rest, news2016+2017+2018 FBNoise filtered backtranslation
@registry.register_problem
class TranslateDeenWmt19NaivepcOs32Fbnoisebtfilt1618(TranslateEndeWmt19):
  pass

# 2x Naively (WMT18) and MS-15M filtered paracrawl, 3x oversampling rest, news2016+2017+2018 FBNoise filtered backtranslation
@registry.register_problem
class TranslateEndeWmt19Msnaivepc15Os32Fbnoisebtfilt1618(TranslateEndeWmt19):
  pass

# 2x Naively (WMT18) and MS-15M filtered paracrawl, 3x oversampling rest, news2016+2017+2018 FBNoise filtered backtranslation
@registry.register_problem
class TranslateDeenWmt19Msnaivepc15Os32Fbnoisebtfilt1618(TranslateEndeWmt19):
  pass

# WMT08-17 test sets
@registry.register_problem
class TranslateEndeWmt19Newstest0817(TranslateEndeWmt19):
  pass

@registry.register_problem
class TranslateDeenWmt19Newstest0817(TranslateEndeWmt19):
  pass

# WMT08-16 test sets
@registry.register_problem
class TranslateEndeWmt19Newstest0816(TranslateEndeWmt19):
  pass

@registry.register_problem
class TranslateDeenWmt19Newstest0816(TranslateEndeWmt19):
  pass



# END WMT19 ------------------------------------------------

# START TEXTNORM --------------------------------------------------

@registry.register_problem
class TextnormEnEnnennopop(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.en"
  @property
  def trg_vocab_file(self):
    return "vocab.en-nen"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TextnormEneopEnnennopop(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.en"
  @property
  def trg_vocab_file(self):
    return "vocab.en-nen"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TextnormEneopEnnen(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.en"
  @property
  def trg_vocab_file(self):
    return "vocab.en-nen"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TextnormCeneopEnnen(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.cen"
  @property
  def trg_vocab_file(self):
    return "vocab.cen-nen"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TextnormCeneopCennen(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.cen"
  @property
  def trg_vocab_file(self):
    return "vocab.cen-nen"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TextnormEneopNen(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.en"
  @property
  def trg_vocab_file(self):
    return "vocab.nen"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

# START GEC19 OSM -------------------------------------------

@registry.register_problem
class TranslateNeenGec19(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.gec19"
  @property
  def trg_vocab_file(self):
    return "vocab.gec19"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TranslateNeenGec19Osm(TranslateNeenGec19):
  pass

@registry.register_problem
class TranslateNeenGec19Aposm(TranslateNeenGec19):
  pass

@registry.register_problem
class TranslateNeenGec19Cposm(TranslateNeenGec19):
  pass

@registry.register_problem
class TranslateNeenGec19Apcposm(TranslateNeenGec19):
  pass

# START GEC19 ------------------------------------------------

# LM

@registry.register_problem
class LanguagemodelEnGec19Unsupervised(LanguagemodelLm1b32k):
  @property
  def vocab_file(self):
    return "vocab.en.gec19.unsupervised"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    return {"targets": text_encoder.TokenTextEncoder(vocab_filename)}

@registry.register_problem
class LanguagemodelEnGec19UnsupervisedChar(LanguagemodelLm1b32k):
  @property
  def vocab_file(self):
    return "vocab.en.gec19.unsupervised.char"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    return {"targets": text_encoder.TokenTextEncoder(vocab_filename)}

# TM

@registry.register_problem
class TranslateNeenGec19Supervised(TranslateEndeWmt32k):
  @property
  def src_vocab_file(self):
    return "vocab.en.gec19.supervised"
  @property
  def trg_vocab_file(self):
    return "vocab.en.gec19.supervised"

  def feature_encoders(self, data_dir):
    src_vocab_filename = os.path.join(data_dir, self.src_vocab_file)
    trg_vocab_filename = os.path.join(data_dir, self.trg_vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(src_vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(trg_vocab_filename)}

@registry.register_problem
class TranslateNeenGec19SupervisedLocness4(TranslateNeenGec19Supervised):
  pass

@registry.register_problem
class TranslateNeenGec19SupervisedLocness4Noid(TranslateNeenGec19Supervised):
  pass

# Backtranslation
@registry.register_problem
class TranslateNeenGec19SupervisedLocness4NoidBt1(TranslateNeenGec19Supervised):
  pass

@registry.register_problem
class TranslateNeenGec19SupervisedLocness4NoidBt3(TranslateNeenGec19Supervised):
  pass

@registry.register_problem
class TranslateNeenGec19SupervisedLocness4NoidBt5(TranslateNeenGec19Supervised):
  pass

@registry.register_problem
class TranslateNeenGec19SupervisedLocness4NoidOs3Bt3(TranslateNeenGec19Supervised):
  pass

@registry.register_problem
class TranslateNeenGec19SupervisedLocness4NoidOs6Bt5(TranslateNeenGec19Supervised):
  pass



@registry.register_problem
class TranslateEnneGec19SupervisedLocness4Noid(TranslateNeenGec19Supervised):
  pass

@registry.register_problem
class TranslateNeenGec19SupervisedLocness8(TranslateNeenGec19Supervised):
  pass

@registry.register_problem
class TranslateNeenGec19SupervisedLocnessonly(TranslateNeenGec19Supervised):
  pass

# END GEC19 ------------------------------------------------


ratio_desired_mean = 3.0

def norm_pdf(x, mu=0.0, sigma=2.0):
  return 1.0 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def _filter_by_ratio_py(inputs, targets):
  global ratio_desired_mean
  u = random.random()
  ratio = float(targets.shape[0]) / float(inputs.shape[0])
  threshold = norm_pdf(ratio, ratio_desired_mean) / norm_pdf(ratio_desired_mean, ratio_desired_mean)
  #print("%f > %f (%f, %f)" % (threshold, u, ratio_desired_mean, ratio))
  return threshold > u

def filter_by_ratio(example):
  #return tf.shape(example["inputs"])[0] > tf.shape(example["targets"])
  return tf.py_func(_filter_by_ratio_py, [example["inputs"],example["targets"]], tf.bool)

@registry.register_problem
class TranslateDeenWmt19BaseLtOld(TranslateEndeWmt19):

  def preprocess_example(self, example, mode, hparams):
    """Runtime preprocessing.

    Return a dict or a tf.Data.Datset.from_tensor_slices (if you want each
    example to turn into multiple).

    Args:
      example: dict, features
      mode: tf.estimator.ModeKeys
      hparams: HParams, model hyperparameters

    Returns:
      dict or Dataset
    """
    example = preprocess_example_common(example, hparams, mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      #example["inputs"] = usr_utils.print_shape(example["inputs"], "inputs", dtype=tf.int64)
      dataset = tf.data.Dataset.from_tensors(example)
      filtered = dataset.filter(filter_by_ratio)
      tf.logging.info("Filter by length!")
      return filtered
    return example



@registry.register_problem
class TranslateEnenGec32k(TranslateEndeWmt32k):
  @property
  def vocab_file(self):
    return "vocab.en.32k.idx"

  def feature_encoders(self, data_dir):
    vocab_filename = os.path.join(data_dir, self.vocab_file)
    return {"inputs": text_encoder.TokenTextEncoder(vocab_filename), 
            "targets": text_encoder.TokenTextEncoder(vocab_filename)}


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

