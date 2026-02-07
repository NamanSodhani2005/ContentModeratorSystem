# Content Moderation System

Content moderation pipeline using supervised NLP + reinforcement learning.

- Supervised models handle text understanding and toxicity signals.
- DQN learns moderation actions (`keep`, `warn`, `remove`, `temp_ban`, `perma_ban`).
- FastAPI backend + React frontend for interactive moderation and feedback.

## Features

- DistilBERT-based text embeddings
- Target-aware toxicity features (target span + hate/offensive/normal scores)
- Supervised toxicity encoder fine-tuning
- DQN policy over moderation actions
- Feedback endpoint for incremental policy updates

## Technology Stack

- Python (training + backend)
- PyTorch (model training and inference)
- Hugging Face Transformers (DistilBERT models/tokenizers)
- Gymnasium (RL environment)
- FastAPI + Uvicorn (backend API)
- React + Vite (frontend)
- NumPy + Pandas (data processing)
- Detoxify (toxicity scoring fallback)

## Repository Structure

```text
ContentModerationSystem/
|-- backend/
|   |-- api/app.py
|   |-- data/
|   |   |-- train.csv                      # required input dataset (not tracked)
|   |   |-- preprocess.py
|   |   |-- augment_stance.py              # optional synthetic augmentation
|   |   `-- ... optional supplementary datasets
|   |-- rl_training/
|   |   |-- train.py
|   |   |-- train_target_span_model.py
|   |   `-- train_toxicity_encoder.py
|   |-- requirements.txt
|   `-- saved_models/                      # generated checkpoints
|-- frontend/
|-- run.py
|-- QUICKSTART.md
`-- README.md
```

## Requirements

- Python 3.9+
- Node.js 16+
- Optional CUDA GPU (recommended)

## Installation

### Backend

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

### Frontend

```bash
cd frontend
npm install
cd ..
```

## Data

Required file:
- `backend/data/train.csv`

Required columns:
- `comment_text`
- `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

Optional supplementary datasets are auto-used if present (HateXplain, Jigsaw variants, Ethos, SBIC, CONAN, etc.).

## Training

### Standard pipeline

```bash
python run.py train
```

Pipeline stages:
1. Target span model (if required target-span dataset trigger exists)
2. Supervised toxicity encoder
3. Preprocessing (`embeddings.npy`, `labels.npy`, feature arrays)
4. DQN training
5. Stance-pair policy gate check

### Full retrain from scratch

```bash
python run.py train --force-target-span --force-toxicity-encoder --force-preprocess
```

### Important CLI flags

```bash
python run.py train --help
```

Common flags:
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

### Custom target-span batch size

`run.py train` does not expose target-span batch size directly.
Run target-span manually when you want custom batch size:

```bash
python backend/rl_training/train_target_span_model.py --batch-size 128 --epochs 5
python run.py train --force-toxicity-encoder --force-preprocess
```

## Serving

Backend + frontend:

```bash
python run.py serve
```

Backend only:

```bash
python run.py serve --backend-only
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`

## API

- `POST /api/moderate`
- `POST /api/feedback`
- `GET /api/metrics`
- `GET /api/examples`