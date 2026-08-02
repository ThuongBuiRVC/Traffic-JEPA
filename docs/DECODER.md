# The graph decoder

The decoder is a post-process. It never trains on real labels and it never replaces the
model as the source of an answer — it only reconciles the model's per-phase answers into a
sequence that is logically consistent. This note records the design and the measurements that
show the model, not the decoder, drives the result.

## Formulation

For each question the frozen model produces a score per answer option (cosine between the
predicted embedding and each option embedding). The decoder combines four log-probability
terms and takes the MAP answer:

1. **Emission** — the model score. This is the primary term.
2. **Answer prior** — `P(answer | question)`, from sim answer frequencies.
3. **Question relation** — `P(answer_dst | answer_src)` between related questions in the same
   scenario (e.g. vehicle-distance ↔ pedestrian-position).
4. **Phase transition** — `P(answer_t | answer_{t-1})` across the ordered event phases
   (prerecognition → recognition → judgement → action → avoidance), MAP-decoded with Viterbi
   so a video's answers form one consistent trajectory.

Every conditional is a probability estimated from sim counts with add-alpha (Lidstone) smoothing:

```
P(b | a) = (count(a, b) + alpha) / (count(a) + alpha * K)
```

`alpha = SMOOTHING_ALPHA` (`traffic_jepa/postprocess/graph_decode.py`), `K` = number of answer
options. An event never seen in sim keeps a small non-zero probability that shrinks as more sim
evidence accumulates — no hardcoded penalty constant. `alpha` was chosen on labeled validation:
accuracy is flat for `alpha <= 1e-3` and degrades for larger `alpha`.

## Keeping the model in scale

The prior/relation/transition terms live in log-probability space, whose magnitude is unrelated
to the model's cosine-score scale, so how far each one may move an answer has to be stated
explicitly. The three terms differ, and only the first two are bounded by the model:

- **Answer prior** is scaled by weights an order of magnitude below the model's own margin
  (`gamma = 0.02`, `gamma_qphase = 0.12`), so it can tip an uncertain answer and little else.
- **Question relation** re-ranks each destination row within `relation_cap` multiples of that
  row's own model-score span (`apply_relation_scores`, default `relation_cap = 2`). It can move an
  answer inside the model's margin but cannot swamp it.
- **Phase transition** is deliberately *not* bounded that way. At each chain node the per-question
  emission margin is small (median 0.174 on labelled validation), so the transition term, which
  spans the full log-probability range of the observed phase sequence, can decide which answer
  wins inside a chain — the trajectory, not the per-question score. This is intended: a predictor
  that answers each phase in isolation has no view of the trajectory, and the sequence prior is
  exactly the evidence it is missing. Its value is measured, not assumed — removing it costs 5.1
  points (88.6 -> 83.5 on real-val).

The write-back keeps the reported scores on the model's scale: the Viterbi MAP answer is lifted
one score span above the row's maximum and every other option keeps its model score. That bounds
the *scores* the next stage sees, not the *decision* the chain makes.

## Why the model is primary, not the decoder

Measured on real-val (11733 labeled questions), current decoder:

### 1. The model alone already carries the result

| Configuration | real-val acc |
|---|---|
| Model alone (no decode) | 0.8128 |
| + answer prior | 0.8164 |
| + question relation | 0.8352 |
| + phase transition (full) | 0.8864 |

The frozen model reaches 0.8128 on its own; the decoder adds +7.4 pp on top of an already
strong model.

### 2. The Viterbi emission is the model score

In the transition decode, the score of choosing option `j` at phase `t` is
`model_score[t][j] + beta * log P(transition)`. When a phase chain is already consistent, every
transition is plausible (`log P ≈ 0`) and the model scores decide. The decoder only departs
from the model when a transition is implausible.

### 3. Edits land on the questions the model is unsure about

The decoder changes 1703 / 11733 = 14.5 % of answers. Those changes are concentrated where the
model has a small margin (top-1 minus top-2 score):

| | median model margin |
|---|---|
| all questions | 0.192 |
| questions the decoder changed | 0.052 |

94 % of the changed answers had a model margin below 0.2. Where the model is confident, the
decoder leaves the answer alone.

The largest term, the phase transition, is exactly what a frozen per-phase model cannot supply on
its own: the model answers each phase in isolation and has no view of the trajectory, so the
transition table only supplies the cross-phase consistency the model is missing.

Knobs on `traffic_jepa/postprocess/graph_decode.py` toggle each term (`--temporal_beta 0` drops the
phase transition, `--alpha 0` the question relation, `--gamma 0` the answer prior) and bound its
scale (`--relation_cap` for the relation term); `scripts/eval_val.sh` prints the real-val accuracy
per category.
