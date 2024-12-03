# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
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

import math
import re
from typing import Dict, List


def log_add(*args) -> float:
    """
    Stable log add
    """
    if all(a == -float("inf") for a in args):
        return -float("inf")
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def tokenize_by_bpe_model(sp, txt, upper=True):
    tokens = []
    # CJK(China Japan Korea) unicode range is [U+4E00, U+9FFF], ref:
    # https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    pattern = re.compile(r"([\u4e00-\u9fff])")
    # Example:
    #   txt   = "你好 ITS'S OKAY 的"
    #   chars = ["你", "好", " ITS'S OKAY ", "的"]
    chars = pattern.split(txt.upper() if upper else txt)
    mix_chars = [w for w in chars if len(w.strip()) > 0]
    for ch_or_w in mix_chars:
        # ch_or_w is a single CJK charater(i.e., "你"), do nothing.
        if pattern.fullmatch(ch_or_w) is not None:
            tokens.append(ch_or_w)
        # ch_or_w contains non-CJK charaters(i.e., " IT'S OKAY "),
        # encode ch_or_w using bpe_model.
        else:
            for p in sp.encode_as_pieces(ch_or_w):
                tokens.append(p)
    return tokens


def tokenize(contexts: List[str], symbol_table: Dict[str, int], bpe_model=None):
    """
    Tokenize and convert contexts into token ids.
    """
    sp = None
    if bpe_model is not None:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)

    context_list = []
    for context in contexts:
        context = context.strip()
        if sp is not None:
            tokens = tokenize_by_bpe_model(sp, context)
        else:
            tokens = list(context.replace(" ", "▁"))

        labels = []
        for ch in tokens:
            if ch in symbol_table:
                labels.append(symbol_table[ch])
            elif "<unk>" in symbol_table:
                labels.append(symbol_table["<unk>"])
        context_list.append(labels)
    return context_list
