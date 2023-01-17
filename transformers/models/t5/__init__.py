# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from ...file_utils import (
    _BaseLazyModule,
    is_flax_available,
    is_sentencepiece_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_t5": ["T5_PRETRAINED_CONFIG_ARCHIVE_MAP", "T5Config"],
}

if is_sentencepiece_available():
    _import_structure["tokenization_t5"] = ["T5Tokenizer"]

if is_tokenizers_available():
    _import_structure["tokenization_t5_fast"] = ["T5TokenizerFast"]

if is_torch_available():
    _import_structure["modeling_t5"] = [
        "T5_PRETRAINED_MODEL_ARCHIVE_LIST",
        "T5EncoderModel",
        "T5ForConditionalGeneration",
        "T5Model",
        "T5PreTrainedModel",
        "load_tf_weights_in_t5",
    ]
    _import_structure["modeling_constrained_t5"] = [
        "T5ForConstrainedConditionalGeneration",
    ]

if is_tf_available():
    _import_structure["modeling_tf_t5"] = [
        "TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFT5EncoderModel",
        "TFT5ForConditionalGeneration",
        "TFT5Model",
        "TFT5PreTrainedModel",
    ]

if is_flax_available():
    _import_structure["modeling_flax_t5"] = [
        "FlaxT5ForConditionalGeneration",
        "FlaxT5Model",
        "FlaxT5PreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config

    if is_sentencepiece_available():
        from .tokenization_t5 import T5Tokenizer

    if is_tokenizers_available():
        from .tokenization_t5_fast import T5TokenizerFast

    if is_torch_available():
        from .modeling_t5 import (
            T5_PRETRAINED_MODEL_ARCHIVE_LIST,
            T5EncoderModel,
            T5ForConditionalGeneration,
            T5Model,
            T5PreTrainedModel,
            load_tf_weights_in_t5,
        )
        from .modeling_constrained_t5 import (
            T5ForConstrainedConditionalGeneration,
        )

    if is_tf_available():
        from .modeling_tf_t5 import (
            TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFT5EncoderModel,
            TFT5ForConditionalGeneration,
            TFT5Model,
            TFT5PreTrainedModel,
        )

    if is_flax_available():
        from .modeling_flax_t5 import FlaxT5ForConditionalGeneration, FlaxT5Model, FlaxT5PreTrainedModel


else:
    import importlib
    import os
    import sys

    class _LazyModule(_BaseLazyModule):
        """
        Module class that surfaces all objects but only performs associated imports when the objects are requested.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        def _get_module(self, module_name: str):
            return importlib.import_module("." + module_name, self.__name__)

    sys.modules[__name__] = _LazyModule(__name__, _import_structure)
