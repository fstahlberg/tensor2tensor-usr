# coding=utf-8
"""Additional problems for standard T2T models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import unicode_literals

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


import sys
import codecs
from collections import defaultdict

sn_i2s = None
sn_s2i = None
sn_bpe_codes = None
sn_separator = '@@'
sn_dropout = None


def _subword_noise_load_files(bpe_codes_path, wmap_path):
  global sn_bpe_codes, sn_i2s, sn_s2i
  tf.logging.info("Loading wmap for subword noise from %s and BPE codes from %s" % (wmap_path, bpe_codes_path))
  # BPE codes
  with codecs.open(bpe_codes_path, encoding='utf-8') as codes_reader:
    sn_bpe_codes = [tuple(item.split()) for item in codes_reader]
  # some hacking to deal with duplicates (only consider first instance)
  sn_bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(sn_bpe_codes)))])
  # WMAP
  sn_i2s = {}
  sn_s2i = {}
  #with open(hparams.sn_wmap) as wmap_reader:
  with codecs.open(wmap_path, encoding='utf-8') as wmap_reader:
    for line in wmap_reader:
      s, i = line.strip().split()
      i = int(i)
      sn_i2s[i] = s
      sn_s2i[s] = i


def get_pairs(word):
    """Return set of symbol pairs in a word.

    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def _subword_noise_encode_py(orig):
    """Splits the subword `orig` into multiple pieces"""
    global sn_bpe_codes, sn_i2s, sn_s2i

    try:
      word = sn_i2s[orig]
      if word[-4:] == u"</w>":
        word = tuple(word[:-4]) + (u'</w>',)
      else:
        word = tuple(word)
      pairs = get_pairs(word)

      while len(pairs) > 1:
        bigram = min(pairs, key = lambda pair: sn_bpe_codes.get(pair, float('inf')))
        if bigram not in sn_bpe_codes:
          break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
          try:
            j = word.index(first, i)
            new_word.extend(word[i:j])
            i = j
          except:
            new_word.extend(word[i:])
            break

          if word[i] == first and i < len(word)-1 and word[i+1] == second:
            new_word.append(first+second)
            i += 2
          else:
            new_word.append(word[i])
            i += 1
        new_word = tuple(new_word)
        word = tuple(new_word)
        pairs = get_pairs(word)

      ret = [sn_s2i[w] for w in word]
      return ret
    except:
      # The try block is just for the case that the BPE segmentation fails
      # due to some kind of UTF-8 fuckery.
      print("BPE segmentation failed!")
    return [orig]


def _subword_noise_encode_noisy_py(word):
  global sn_dropout
  if word < 5 or random.random() > sn_dropout:
    # Leave special symbols
    return [word]
  subwords = _subword_noise_encode_py(word)
  noisy_subwords = []
  for subword in subwords:
    noisy_subwords.extend(_subword_noise_encode_noisy_py(subword))
  return noisy_subwords


def _subword_noise_py(labels):
  noisy_labels = []
  for i in labels:
    noisy_labels.extend(_subword_noise_encode_noisy_py(i))
  return np.array(noisy_labels, dtype=np.int64)


@registry.register_problem
class TranslateEndeWmt19BaseSn(TranslateEndeWmt32k):

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

  def preprocess_example(self, example, mode, hparams):
    global sn_i2s, sn_s2i, sn_dropout
    if sn_i2s is None:
      sn_dropout = hparams.sn_dropout
      _subword_noise_load_files(hparams.sn_bpe_codes, hparams.sn_wmap)
    example = preprocess_example_common(example, hparams, mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      tf.logging.info("Applying subword level noise to inputs!")
      example["inputs"] = tf.py_func(_subword_noise_py, [example["inputs"]], tf.int64)
      example["inputs"].set_shape([None])
      #example["targets"] = tf.py_func(_subword_noise_py, [example["targets"]], tf.int64)
      #example["targets"].set_shape([None])
    return example


# Naively (WMT18) 2x filtered paracrawl, 3x oversampling rest, news2016+2017+2018 FBNoise filtered backtranslation
@registry.register_problem
class TranslateEndeWmt19NaivepcOs32Fbnoisebtfilt1618Sn(TranslateEndeWmt19BaseSn):
  pass

