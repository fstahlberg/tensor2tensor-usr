# coding=utf-8
"""Problem specifications for layer-by-layer models."""


from tensor2tensor.data_generators.wmt import TranslateEndeWmt32k
from tensor2tensor.utils import registry


@registry.register_problem
class TranslateEndeWmt32kLayerbylayer(TranslateEndeWmt32k):
  """Problem spec for WMT En-De translation for layer-by-layer models."""

  def generator(self, data_dir, tmp_dir, train):
    """Data generation outside of T2T."""
    raise NotImplementedError
