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

from .context_graph import ContextState
from .utils import log_add


class PrefixScore:
    """For CTC prefix beam search"""

    def __init__(
        self,
        s: float = float("-inf"),
        ns: float = float("-inf"),
        v_s: float = float("-inf"),
        v_ns: float = float("-inf"),
        context_state: ContextState = None,
        context_score: float = 0.0,
    ):
        self.s = s  # blank_ending_score
        self.ns = ns  # none_blank_ending_score
        self.v_s = v_s  # viterbi blank ending score
        self.v_ns = v_ns  # viterbi none blank ending score
        self.cur_token_prob = float("-inf")  # prob of current token
        self.times_s = []  # times of viterbi blank path
        self.times_ns = []  # times of viterbi none blank path
        self.context_state = context_state
        self.context_score = context_score
        self.has_context = False
        self.token_probs = []  # The maximum probability of each token

    def score(self):
        return log_add(self.s, self.ns)

    def viterbi_score(self):
        return self.v_s if self.v_s > self.v_ns else self.v_ns

    def times(self):
        return self.times_s if self.v_s > self.v_ns else self.times_ns

    def total_score(self):
        return self.score() + self.context_score
