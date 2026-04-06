# Import backends to trigger registry decoration
from averyml.training.backends import hf_trainer as _hf  # noqa: F401
from averyml.training.backends import torchtune_trainer as _torchtune  # noqa: F401
