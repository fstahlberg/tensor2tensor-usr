# coding=utf-8
"""Layer-by-layer model definitions."""


from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from usr.models.tfnmt.nmt.nmt import model as tfnmt_model
from usr.models.tfnmt.nmt.nmt import model_helper
from usr.models.tfnmt.nmt.nmt import attention_model
from tensor2tensor.layers import common_layers
from usr import utils as usr_utils
from usr.models.tfnmt.nmt.nmt.utils import iterator_utils
from usr.models.tfnmt import hparams_helper
import collections

import tensorflow as tf

@registry.register_model
class TFNmtLayerbylayer(t2t_model.T2TModel):
  """Layer-by-layer TF-NMT models."""

  def model_fn_body(self, features):
    hparams = self._hparams
    if not hparams.attention or hparams.attention_architecture != "standard":
      raise ValueError("Layer-by-layer version of TF-NMT only available for "
                       "the 'standard' attention model architecture.")
    inputs, inputs_length = usr_utils.get_feature_with_length(
        features, "inputs")
    target_roots, target_roots_length = usr_utils.get_feature_with_length(
        features, "target_roots")
    targets, targets_length = usr_utils.get_feature_with_length(
        features, "targets")
    # We need to do +1 for inference since get_feature_with_length()
    # may not have direct access to sequence lengths and returns
    # a length of 0 for the first inference step. 
    if hparams.mode == tf.contrib.learn.ModeKeys.INFER:
      targets_length = targets_length + 1
    # input lengths of 0 breaks things
    inputs_length = tf.maximum(inputs_length, 1)
    target_roots_length = tf.maximum(target_roots_length, 1)

    # Shift targets right to use them as input
    targets = common_layers.shift_right_3d(targets)

    # Manage POP signals
    if hparams.target_root_attention == "pop":
        raw_targets = tf.squeeze(tf.squeeze(
            features["raw_targets"], axis=2), axis=2)
        targets_is_pop = tf.equal(raw_targets, hparams.pop_id)
    else:
        targets_is_pop = None

    iterator = TFNmtLayerbylayerInput(
      initializer=None,
      source=inputs,
      target_input=targets,
      target_input_is_pop=targets_is_pop,
      target_output=None, # Loss is computed in T2T
      target_root=target_roots,
      source_sequence_length=inputs_length,
      target_sequence_length=targets_length,
      target_root_sequence_length=target_roots_length)
    tfnmt_model = TFNmtLayerbylayerModel(
      hparams_helper.convert_to_tfnmt_hparams(hparams),
      iterator=iterator,
      mode=tf.contrib.learn.ModeKeys.EVAL,  # We use eval graph for training
      source_vocab_table=FakeVocabTable(),
      target_vocab_table=FakeVocabTable())
    decoder_output = tfnmt_model.logits
    return tf.expand_dims(decoder_output, axis=2)


class TFNmtLayerbylayerInput(collections.namedtuple(
                                 "TFNmtLayerbylayerInput", 
                                 ("initializer",
                                  "source",
                                  "target_input",
                                  "target_input_is_pop",
                                  "target_output",
                                  "target_root",
                                  "source_sequence_length",
                                  "target_sequence_length",
                                  "target_root_sequence_length")),
                             iterator_utils.BatchedInput):
    pass # We inherit from BatchedInput to keep nmt.Model happy.


class IdentityAttentionMechanism(tf.contrib.seq2seq.LuongAttention):
  """This is a very simple AttentionMechanism implementation which realizes the
  identity alignment matrix: initial_alignments() produces the first unit
  vector (1,0,0,...). New alignments are computed from previous alignments by
  shifting them right by one. Note that this deliberately leads to an error
  downstream since it produces a zero vector after the 1 is shifted through the
  entire alignment. The IdentityAttentionMechanism is only working correctly if
  the number of time steps is lower or equal the alignment_size.
  """

  def initial_alignments(self, batch_size, dtype):
    """This is copied from _BaseMonotonicAttentionMechanism in tf.contrib."""
    return tf.one_hot(tf.zeros((batch_size,), dtype=tf.int32), 
                      self._alignments_size, dtype=dtype)

  def __call__(self, unused_query, previous_alignments):
    """Compute next alignments by shifting previous_alignments."""
    return tf.pad(previous_alignments, [[0, 0], [1, 0]])[:,:-1]


def _create_attention_mechanism(attention_option, num_units, memory,
                                memory_length, target_input_is_pop):
  """Create attention mechanism based on the attention_option. Passes
  through to attention_model.create_attention_mechanism() if the
  attention_option is not POP. Construct the attention mechanism by
  repeating entries in memory and using the IdentityAttentionMechanism
  if attention_option is POP.
  """
  if attention_option != "pop":
    return attention_model.create_attention_mechanism(
        attention_option, num_units, memory, memory_length)
  # We need to use an offset of 2:
  # - One since first alignment of the IdentityAttentionMechanism is 0,1,0,0,..
  # - One to delay the input s.t. target root is updated *after* POP
  extended_memory = usr_utils.expand_memory_by_pop(
      target_input_is_pop, memory, offset=2)
  return IdentityAttentionMechanism(num_units, extended_memory)


class TFNmtLayerbylayerModel(tfnmt_model.Model):
  """Sequence-to-sequence dynamic model with attention.

  This class follows nmt.attention_model.AttentionModel, but adds the 
  additional input target_roots to the decoder RNN.
  """

  def __init__(self,
               hparams,
               mode,
               iterator,
               source_vocab_table,
               target_vocab_table,
               reverse_target_vocab_table=None,
               scope=None,
               single_cell_fn=None):
    super(TFNmtLayerbylayerModel, self).__init__(
        hparams=hparams,
        mode=mode,
        iterator=iterator,
        source_vocab_table=FakeVocabTable(), # Not used
        target_vocab_table=FakeVocabTable(), # Not used
        reverse_target_vocab_table=reverse_target_vocab_table,
        scope=scope,
        single_cell_fn=single_cell_fn)

  def _build_target_root_encoder(self, hparams):
    """Builds the encoder for the target roots."""
    num_layers = hparams.num_layers
    num_residual_layers = hparams.num_residual_layers
    iterator = self.iterator
    with tf.variable_scope("target_root_encoder") as scope:
      dtype = scope.dtype
      encoder_emb_inp = iterator.target_root

      # Encoder_outpus: [max_time, batch_size, num_units]
      if hparams.target_root_encoder_type == "uni":
        cell = self._build_encoder_cell(
            hparams, num_layers, num_residual_layers)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell,
            encoder_emb_inp,
            dtype=dtype,
            sequence_length=iterator.target_root_sequence_length,
            time_major=self.time_major)
      elif hparams.target_root_encoder_type == "id":
        encoder_outputs, encoder_state = encoder_emb_inp, encoder_emb_inp 
      elif hparams.target_root_encoder_type == "bi":
        num_bi_layers = int(num_layers / 2)
        num_bi_residual_layers = int(num_residual_layers / 2)

        encoder_outputs, bi_encoder_state = (
            self._build_bidirectional_rnn(
                inputs=encoder_emb_inp,
                sequence_length=iterator.target_root_sequence_length,
                dtype=dtype,
                hparams=hparams,
                num_bi_layers=num_bi_layers,
                num_bi_residual_layers=num_bi_residual_layers))

        if num_bi_layers == 1:
          encoder_state = bi_encoder_state
        else:
          # alternatively concat forward and backward states
          encoder_state = []
          for layer_id in range(num_bi_layers):
            encoder_state.append(bi_encoder_state[0][layer_id])  # forward
            encoder_state.append(bi_encoder_state[1][layer_id])  # backward
          encoder_state = tuple(encoder_state)
      else:
        raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)

    return encoder_outputs, encoder_state

  def build_graph(self, hparams, scope=None):
    """Extends super class implementation by calling 
    `_build_target_root_encoder(hparams)`.
    """
    tf.logging.info("Creating %s graph ..." % self.mode)
    dtype = tf.float32

    with tf.variable_scope(scope or "dynamic_seq2seq", dtype=dtype):
      # Encoder
      encoder_outputs, encoder_state = self._build_encoder(hparams)

      # Target roots encoder
      target_root_encoder_outputs, target_root_encoder_state = \
        self._build_target_root_encoder(hparams)

      ## Decoder
      # encoder_outputs and encoder_state is a 2-tuple of tensors, not
      # plain tensors like in the original implementation to pass through
      # the target root encoder.
      logits, sample_id, final_context_state = self._build_decoder(
          (encoder_outputs, target_root_encoder_outputs), 
          (encoder_state, target_root_encoder_state), 
          hparams)

      return logits, None, final_context_state, sample_id

  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Extends the super class implementation by the additional input 
    from the target roots.
    """
    encoder_outputs, target_root_encoder_outputs = encoder_outputs
    encoder_state, target_root_encoder_state = encoder_state
    target_root_sequence_length = self.iterator.target_root_sequence_length

    attention_option = hparams.attention
    target_root_attention_option = hparams.target_root_attention
    attention_architecture = hparams.attention_architecture

    if attention_architecture != "standard":
      raise ValueError(
          "Unknown attention architecture %s" % attention_architecture)

    num_units = hparams.num_units
    num_layers = hparams.num_layers
    num_residual_layers = hparams.num_residual_layers
    num_gpus = hparams.num_gpus
    beam_width = hparams.beam_width
    dtype = tf.float32
    batch_size = self.batch_size

    att_mechanisms = []
    # Attention over source sentence
    att_mechanisms.append(attention_model.create_attention_mechanism(
        attention_option, num_units, encoder_outputs, source_sequence_length))
    # Attention over target roots
    att_mechanisms.append(_create_attention_mechanism(
        target_root_attention_option, num_units, target_root_encoder_outputs,
        target_root_sequence_length, self.iterator.target_input_is_pop))

    cell = model_helper.create_rnn_cell(
        unit_type=hparams.unit_type,
        num_units=num_units,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        num_gpus=num_gpus,
        mode=self.mode,
        single_cell_fn=self.single_cell_fn)

    cell = tf.contrib.seq2seq.AttentionWrapper(
        cell,
        att_mechanisms,
        attention_layer_size=[num_units/2, num_units/2],
        alignment_history=False,
        name="attention")

    # fstahlberg: We don't need that in T2T (?)
    # TODO(thangluong): do we need num_layers, num_gpus?
    #cell = tf.contrib.rnn.DeviceWrapper(cell,
    #                                    model_helper.get_device_str(
    #                                        num_layers - 1, num_gpus))

    if hparams.pass_hidden_state:
      decoder_initial_state = cell.zero_state(batch_size, dtype).clone(
          cell_state=encoder_state)
    else:
      decoder_initial_state = cell.zero_state(batch_size, dtype)

    return cell, decoder_initial_state


class FakeVocabTable(object):
  def lookup(self, unused_arg):
    return 99999999


