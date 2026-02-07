# Content Moderation System

End-to-end moderation pipeline with:
- DistilBERT embeddings
- target-aware toxicity features
- supervised toxicity encoder (multi-label)
- DQN policy for final moderation actions (`keep`, `warn`, `remove`, `temp_ban`, `perma_ban`)

## What This Repository Contains

- `backend/data/preprocess.py`: builds `embeddings.npy`, `labels.npy`, and optional target/toxicity feature arrays
- `backend/rl_training/train_target_span_model.py`: trains target span + 3-class toxicity model
- `backend/rl_training/train_toxicity_encoder.py`: fine-tunes DistilBERT toxicity encoder
- `backend/rl_training/train.py`: offline DQN training
- `backend/api/app.py`: FastAPI inference + feedback endpoint
- `run.py`: unified train/serve entrypoint

## Project Layout

```text
ContentModerationSystem/
|-- backend/
|   |-- api/
|   |   `-- app.py
|   |-- data/
|   |   |-- train.csv                     # required (not tracked)
|   |   |-- dataset.json                  # optional (HateXplain)
|   |   |-- archive/labeled_data.csv      # optional for target-span stage trigger in run.py
|   |   |-- preprocess.py
|   |   |-- augment_stance.py             # optional synthetic stance augmenter
|   |   `-- ... other optional datasets
|   |-- rl_training/
|   |   |-- train.py
|   |   |-- train_target_span_model.py
|   |   |-- train_toxicity_encoder.py
|   |   |-- agents/
|   |   |-- environment/
|   |   `-- models/
|   |-- requirements.txt
|   `-- saved_models/                     # generated checkpoints
|-- frontend/
|   |-- src/
|   |-- package.json
|   `-- vite.config.js
|-- run.py
|-- QUICKSTART.md
`-- README.md
```

## Prerequisites

- Python 3.9+
- Node.js 16+
- CUDA-capable GPU optional but recommended

## Setup

### 1. Python environment

Windows (PowerShell):

```powershell
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Frontend deps

```bash
cd frontend
npm install
cd ..
```

## Data Requirements

Required:
- `backend/data/train.csv` with columns:
  - `comment_text`
  - `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

Optional datasets used automatically if present:
- `backend/data/dataset.json` (HateXplain)
- `backend/data/archive/labeled_data.csv`
- `backend/data/hate-speech-and-offensive-language.csv`
- `backend/data/jigsaw-toxic-comment-classification-challenge/train.csv`
- `backend/data/Ethos_Dataset_Binary.csv`
- `backend/data/Ethos_Dataset_Multi_Label.csv`
- `backend/data/en_dataset_with_stop_words.csv`
- `backend/data/measuring_hate_speech.csv`
- `backend/data/SBIC.v2.agg.trn.csv`
- `backend/data/archive2/test (1).csv`
- `backend/data/Multitarget-CONAN.csv`
- `backend/data/comments.txt` (improves lexicon coverage)

Optional synthetic stance augmentation:

```bash
python backend/data/augment_stance.py
```

## Training

### One command pipeline

```bash
python run.py train
```

Pipeline order:
1. Target span model (if `backend/data/archive/labeled_data.csv` exists)
2. Supervised toxicity encoder
3. Preprocess embeddings/features
4. DQN training
5. Policy stance-pair gate check

### Full retrain from scratch (without deleting files)

```bash
python run.py train --force-target-span --force-toxicity-encoder --force-preprocess
```

### Common flags

```bash
python run.py train --help
```

Important flags:
- `--force-target-span`
- `--force-toxicity-encoder`
- `--skip-toxicity-encoder`
- `--toxicity-epochs`
- `--toxicity-batch-size`
- `--toxicity-lr`
- `--force-preprocess`
- `--skip-existing`
- `--skip-dqn-pretrain`
- `--allow-stance-fail`
- `--download-stance-data`

### Why it may train for only 2 epochs

`run.py train` defaults toxicity encoder to:
- `--toxicity-epochs 2`

So seeing `Epoch 1/2` is expected unless you override it.

Example:

```bash
python run.py train --force-toxicity-encoder --toxicity-epochs 5 --toxicity-batch-size 128
```

### Custom target-span batch size (manual stage)

`run.py train` currently uses internal defaults for target-span stage.
If you want custom target-span batch size, run that stage directly:

```bash
python backend/rl_training/train_target_span_model.py --batch-size 128 --epochs 5
```

Then continue pipeline while skipping target-span retrain:

```bash
python run.py train --force-toxicity-encoder --force-preprocess
```

## Serve

Backend + frontend:

```bash
python run.py serve
```

Backend only:

```bash
python run.py serve --backend-only
```

Endpoints:
- `POST /api/moderate`
- `POST /api/feedback`
- `GET /api/metrics`
- `GET /api/examples`

## GPU Checks

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

If CUDA is visible but utilization is low, that does not automatically mean misconfiguration. Throughput also depends on dataloader speed, sequence length, and model size.

## Troubleshooting

### `run.py train ... --batch_size 128` fails

Use the correct flag names:
- `run.py train`: `--toxicity-batch-size`
- `train_target_span_model.py`: `--batch-size`

`--batch_size` is not a valid argument in this repo.

### `libcudnn.so.9` or torch import errors on remote

Use a clean venv and reinstall dependencies in that venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
export PYTHONNOUSERSITE=1
pip install -U pip
pip install -r backend/requirements.txt
```

### Target-span data counts look too low

That usually means optional datasets are missing on that machine. The trainer will silently skip missing files and continue.

## Notes

- `backend/data/augment_stance.py` is synthetic data augmentation (templated examples).
- The trained models are still learned neural models at inference time; they are not runtime keyword matchers.

## License

Use according to your project/license policy.
