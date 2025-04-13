# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2023 Tsinghua Univ. (authors: Xingchen Song)
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
from collections import defaultdict
from typing import Dict, List

import torch

from .context_graph import ContextGraph
from .prefix_score import PrefixScore
from .utils import log_add


class CTCDecoder:
    def __init__(
        self,
        contexts: List[str] = None,
        symbol_table: Dict[str, int] = None,
        bpe_model: str = None,
        context_score: float = 6.0,
        blank_id: int = 0,
    ):
        self.context_graph = None
        if contexts is not None:
            self.context_graph = ContextGraph(contexts, symbol_table, bpe_model, context_score)
        self.blank_id = blank_id
        self.cur_t = 0
        self.cur_hyps = []
        self.reset()

    def reset(self):
        self.cur_t = 0
        context_root = None if self.context_graph is None else self.context_graph.root
        self.cur_hyps = [(tuple(), PrefixScore(s=0.0, v_s=0.0, context_state=context_root))]

    def copy_context(self, prefix_score, next_score):
        # perfix not changed, copy the context from prefix
        if self.context_graph is not None and not next_score.has_context:
            next_score.context_score = prefix_score.context_score
            next_score.context_state = prefix_score.context_state
            next_score.has_context = True

    def update_context(self, prefix_score, next_score, word_id):
        if self.context_graph is not None and not next_score.has_context:
            context_score, context_state = self.context_graph.forward_one_step(prefix_score.context_state, word_id)
            next_score.context_score = prefix_score.context_score + context_score
            next_score.context_state = context_state
            next_score.has_context = True

    def backoff_context(self):
        if self.context_graph is not None:
            # We should backoff the context score/state when the context is
            # not fully matched at the last time.
            for i, hyp in enumerate(self.cur_hyps):
                score, new_state = self.context_graph.finalize(hyp[1].context_state)
                self.cur_hyps[i][1].context_score = score
                self.cur_hyps[i][1].context_state = new_state

    def ctc_greedy_search(self, ctc_probs: torch.Tensor, is_last: bool = False, return_probs: bool = False):
        results = self.ctc_prefix_beam_search(ctc_probs, 1, is_last, return_probs)
        if is_last:
            self.reset()
        if return_probs:
            return {"tokens": results["tokens"][0], "times": results["times"][0], "probs": results["probs"][0]}
        return {"tokens": results["tokens"][0], "times": results["times"][0]}

    def ctc_prefix_beam_search(
        self, ctc_probs: torch.Tensor, beam_size: int, is_last: bool = False, return_probs: bool = False
    ):
        for logp in ctc_probs:
            self.cur_t += 1
            # key: prefix, value: PrefixScore
            next_hyps = defaultdict(lambda: PrefixScore())
            # 1. First beam prune: select topk best
            logp, indices = logp.topk(beam_size)  # (beam_size,)
            for prob, u in zip(logp.tolist(), indices.tolist()):
                for prefix, prefix_score in self.cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if u == self.blank_id:  # blank
                        next_score = next_hyps[prefix]
                        next_score.s = log_add(next_score.s, prefix_score.score() + prob)
                        next_score.v_s = prefix_score.viterbi_score() + prob
                        next_score.times_s = prefix_score.times().copy()
                        if return_probs:
                            next_score.token_probs = prefix_score.token_probs.copy()
                        # perfix not changed, copy the context from prefix
                        self.copy_context(prefix_score, next_score)
                    elif u == last:
                        # Update *uu -> *u;
                        next_score1 = next_hyps[prefix]
                        next_score1.ns = log_add(next_score1.ns, prefix_score.ns + prob)
                        if next_score1.v_ns < prefix_score.v_ns + prob:
                            next_score1.v_ns = prefix_score.v_ns + prob
                            if next_score1.cur_token_prob < prob:
                                next_score1.cur_token_prob = prob
                                next_score1.times_ns = prefix_score.times_ns.copy()
                                next_score1.times_ns[-1] = self.cur_t
                        if return_probs:
                            next_score1.token_probs = prefix_score.token_probs.copy()
                            next_score1.token_probs[-1] = max(next_score1.token_probs[-1], prob)
                        self.copy_context(prefix_score, next_score1)
                        # Update *u-u -> *uu, - is for blank
                        n_prefix = prefix + (u,)
                        next_score2 = next_hyps[n_prefix]
                        next_score2.ns = log_add(next_score2.ns, prefix_score.s + prob)
                        if next_score2.v_ns < prefix_score.v_s + prob:
                            next_score2.v_ns = prefix_score.v_s + prob
                            next_score2.cur_token_prob = prob
                            next_score2.times_ns = prefix_score.times_s.copy()
                            next_score2.times_ns.append(self.cur_t)
                        if return_probs:
                            next_score2.token_probs = prefix_score.token_probs.copy()
                            next_score2.token_probs.append(prob)
                        self.update_context(prefix_score, next_score2, u)
                    else:
                        n_prefix = prefix + (u,)
                        next_score = next_hyps[n_prefix]
                        next_score.ns = log_add(next_score.ns, prefix_score.score() + prob)
                        if next_score.v_ns < prefix_score.viterbi_score() + prob:
                            next_score.v_ns = prefix_score.viterbi_score() + prob
                            next_score.cur_token_prob = prob
                            next_score.times_ns = prefix_score.times().copy()
                            next_score.times_ns.append(self.cur_t)
                        if return_probs:
                            next_score.token_probs = prefix_score.token_probs.copy()
                            next_score.token_probs.append(prob)
                        self.update_context(prefix_score, next_score, u)

            # 2. Second beam prune
            next_hyps = sorted(next_hyps.items(), key=lambda x: x[1].total_score(), reverse=True)
            self.cur_hyps = next_hyps[:beam_size]

        cur_hyps = self.cur_hyps
        if is_last:
            self.backoff_context()
            self.reset()

        response = {"tokens": [list(y[0]) for y in cur_hyps], "times": [y[1].times() for y in cur_hyps]}
        if return_probs:
            response["probs"] = [[math.exp(p) for p in y[1].token_probs] for y in cur_hyps]
        return response
