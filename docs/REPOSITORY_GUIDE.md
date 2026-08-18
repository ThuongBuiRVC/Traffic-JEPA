# Repository guide

This guide is a code-oriented companion to the main README. It explains where each stage lives,
which artifacts cross stage boundaries, and which entrypoint to use when inspecting or extending
the pipeline.

## Pipeline at a glance

The repository has two submission stages:

1. **Answer prediction** scores every multiple-choice question from video and question context,
   then reconciles related answers with the graph/temporal decoder.
2. **Caption generation** turns the decoded answers into pedestrian and vehicle descriptions,
   optionally grounding the generation with sampled video frames.

The top-level inference command is:

```bash
bash scripts/inference.sh
```

It runs `scripts/04_submit_test.sh` followed by `scripts/05_caption.sh`. The caption stage therefore
depends on the final VQA submission, not on the raw model predictions.

## Module map

| Area | Main modules | Responsibility |
|---|---|---|
| Data preparation | `traffic_jepa/data/build_manifest.py` | Select videos and windows, sample clips, and build simulation manifests. |
| Frozen features | `traffic_jepa/data/preprocess.py` | Cache V-JEPA latent grids and EmbeddingGemma text vectors. |
| Dataset | `traffic_jepa/data/wts_cached.py` | Load cached latents, option vectors, labels, and question metadata. |
| VQA model | `traffic_jepa/modeling/traffic_jepa_model.py` | Fuse visual, phase, category, and question tokens and predict an answer embedding. |
| Predictor backbone | `traffic_jepa/modeling/predictor.py` | Adapt the selected Llama layers for bidirectional attention. |
| VQA training | `traffic_jepa/training/train.py` | Train with answer-embedding contrastive loss and write checkpoints. |
| Test scoring | `traffic_jepa/inference/predict.py` | Build public-test jobs, encode clips, and write raw option scores. |
| VQA orchestration | `traffic_jepa/inference/submit.py` | Run model scoring and deterministic post-processing in sequence. |
| Decoder | `traffic_jepa/postprocess/graph_decode.py` | Apply simulation priors, question relations, and phase transitions. |
| Text captions | `traffic_jepa/captioning/` | Build facts-only LoRA data, train the adapter, and generate captions. |
| Multimodal captions | `traffic_jepa/captioning_mm/` | Extract frames and train or run the facts-plus-frames variant. |

## VQA data flow

During training, SynWTS annotations and videos are converted into 16-frame clips. Preprocessing
keeps eight frames from each clip and writes a V-JEPA latent grid for every unique clip. It also
embeds every unique answer option with EmbeddingGemma. Training loads these cached tensors rather
than repeatedly running either frozen encoder.

The model projects V-JEPA features into the predictor's hidden dimension and prepends them to phase,
category, and tokenized-question embeddings. The bidirectional predictor produces one pooled
768-dimensional vector. Cosine similarity against the option vectors supplies the model scores.

At public-test time, `predict.py` performs the same visual and option encoding online. It writes both
a letter-only raw submission and a scored JSONL file. The decoder consumes the scored JSONL and the
simulation index, then writes the final VQA submission.

## Caption data flow

The caption stage joins each decoded answer letter back to its answer text in the public question
file. Facts are grouped by scenario and event phase before being sent to Qwen3-VL.

Three modes are available:

| Mode | Inputs | Adapter |
|---|---|---|
| `lora` | Decoded VQA facts | Text-only caption LoRA |
| `mm` | Decoded facts and extracted frames | Multimodal caption LoRA |
| `base` | Decoded VQA facts | No adapter |

Generation uses an OpenAI-compatible local server by default. The in-process Transformers backend
is available when `CAPTION_NO_SERVE=1` is set.

## Generated artifacts

| Path | Producer | Purpose |
|---|---|---|
| `data/processed/sim_qa_vljepa16/` | `scripts/01_manifest_sim.sh` | Simulation clips, boxes, and manifests. |
| `data/processed/cache_vljepa16_8f/` | `scripts/02_preprocess.sh` | Cached latent grids, text vectors, metadata, and simulation index. |
| `runs/traffic_jepa_world_model/model_latest.pt` | `scripts/03_train.sh` | Final VQA predictor weights. |
| `submissions/submission_raw.json` | VQA scorer | Model-only answer letters before decoding. |
| `submissions/submission_raw_scored.jsonl` | VQA scorer | Per-option scores and metadata used by the decoder. |
| `submissions/submission_final.json` | Decoder | Final SubTask2 submission and caption-stage input. |
| `submissions/submission_final_decoded.jsonl` | Decoder | Debug record showing decoder decisions. |
| `submissions/caption_submission.json` | Caption generator | Facts-only or base SubTask1 submission. |
| `submissions/caption_submission_mm.json` | Multimodal generator | Facts-plus-frames SubTask1 submission. |

Large generated data, model weights, and submissions are intentionally excluded from the source
tree. The empty directories under `data/` document the expected layout, while the small decoder
index in `checkpoints/` is included so released weights can be used without preprocessing SynWTS.

## Configuration boundaries

- `configs/train_args.json` records the default VQA architecture and training settings.
- A checkpoint must have its matching `run_args.json` beside it; inference uses that file to rebuild
  the architecture exactly.
- `configs/route.json` maps each public-test question template to a semantic category and preferred
  camera view.
- `scripts/env.sh` centralizes paths, model identifiers, caption settings, and decoder weights.

See `docs/DECODER.md` for the decoder formulation, score scaling, and validation ablations.
