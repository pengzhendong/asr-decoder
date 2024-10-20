# asr-decoder

CTC decoder with hotwords for ASR models. Supports `ctc_greedy_search` and `ctc_prefix_bream_search` decoding methods.

## Usage

- ctc greedy search

``` python
from asr_decoder import CTCDecoder

decoder = CTCDecoder()
# ctc_probs: (sequence_length, vocab_size)
result = decoder.ctc_greedy_search(ctc_probs, is_last=True)
```

- ctc prefix beam search (with hotwords)

``` python
from asr_decoder import CTCDecoder

hotwords = ["停止"]
decoder = CTCDecoder(hotwords, symbol_table, bpemodel)
# ctc_probs: (sequence_length, vocab_size)
result = decoder.ctc_prefix_beam_search(ctc_probs, beam_size=3, is_last=True)
```
