# Content Moderation RL System

An end-to-end content moderation stack that applies a DQN policy over DistilBERT embeddings with target-aware toxicity features, served via FastAPI and surfaced in a React UI. The system outputs discrete moderation actions (keep/warn/remove/temp_ban/perma_ban) with confidence, reasoning, and Q-value alternatives.

## Features

- **Deep Q-Network (DQN)** agent with feedforward policy network
- **Target-aware features** from a target span toxicity model (target presence + hate/offensive/normal probs)
- **Stance-aware moderation**: distinguishes anti-hate speech ("I hate nazis") from pro-hate speech ("I love nazis")
- **Continuous alignment reward** with quadratic penalty for mismatched action severity
- **Balanced sampling + reward shaping** to reduce toxic under-moderation
- **Explainable AI**: Q-value transparency and natural language reasoning
- **Real-time moderation** via FastAPI backend
- **Interactive React UI** with toxicity breakdown, alternative actions, and feedback loop
- **Online feedback updates** logged to JSONL for incremental policy tuning

## Tech Stack

### Backend
- **RL Framework**: PyTorch + Gymnasium
- **API**: FastAPI + Uvicorn
- **NLP**: Transformers (DistilBERT), Detoxify
- **Target span model**: DistilBERT + weak supervision + HateXplain data
- **Data**: Pandas, NumPy

### Frontend
- **Framework**: React + Vite
- **Styling**: Custom CSS (minimal monochrome)
- **HTTP**: Axios
- **Icons**: Lucide React

## Project Structure

```
content-moderation-rl/
|-- backend/
|   |-- data/
|   |   |-- train.csv                 # Your dataset (not tracked)
|   |   |-- dataset.json              # HateXplain dataset (20k posts)
|   |   |-- augment_stance.py         # Counterfactual stance augmentation
|   |   |-- download_stance_data.py   # Download external stance datasets
|   |   |-- preprocess.py             # Embeddings + feature generation
|   |   |-- target_lexicon.json       # Generated lexicon (optional)
|   |   `-- feedback.jsonl            # Runtime feedback log
|   |-- rl_training/
|   |   |-- environment/
|   |   |   `-- forum_env.py          # Gymnasium environment
|   |   |-- agents/
|   |   |   `-- dqn_agent.py          # DQN agent + replay buffer
|   |   |-- models/
|   |   |   |-- policy_network.py     # Feedforward Q-network
|   |   |   `-- target_span_model.py  # Target span toxicity model
|   |   |-- train.py                  # DQN training loop
|   |   `-- train_target_span_model.py  # Target span model training
|   |-- api/
|   |   `-- app.py                    # FastAPI server
|   |-- saved_models/                 # Trained checkpoints
|   `-- requirements.txt
|-- frontend/
|   |-- src/
|   |   |-- components/
|   |   |   |-- CommentInput.jsx
|   |   |   `-- ModerationResult.jsx
|   |   |-- App.jsx
|   |   `-- api.js
|   |-- package.json
|   `-- vite.config.js
|-- run.py                            # Unified runner (train + serve)
`-- README.md
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- Dataset CSV at `backend/data/train.csv` (for training)

### 1. Prepare Data

Place a CSV at `backend/data/train.csv` with a `comment_text` column and toxicity label columns
(`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).

Optional datasets for target-aware training (train_target_span_model.py will use any it finds):
- `backend/data/dataset.json` (HateXplain — primary target/stance dataset)
- `backend/data/archive/labeled_data.csv` (hate/offensive labels)
- `backend/data/Ethos_Dataset_Binary.csv`, `backend/data/Ethos_Dataset_Multi_Label.csv`
- `backend/data/SBIC.v2.agg.trn.csv`
- `backend/data/measuring_hate_speech.csv`
- `backend/data/hate-speech-and-offensive-language.csv`
- `backend/data/en_dataset_with_stop_words.csv`
- `backend/data/archive2/test (1).csv` (HateCheck)
- `backend/data/jigsaw-toxic-comment-classification-challenge/train.csv`

Optional stance augmentation (adds counterfactual anti-hate/pro-hate examples):

```bash
python backend/data/augment_stance.py
```

### 2. Backend Setup

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Preprocess data and generate embeddings/features (~30 minutes)
python backend/data/preprocess.py
```

Preprocessing generates `embeddings.npy`, `labels.npy`, `target_features.npy`, and
`target_toxicity.npy`. If you retrain the target span model, rerun preprocessing
to regenerate features.

### 3. Train the Model

Recommended one-shot pipeline (target span model -> preprocess -> DQN -> stance check):

```bash
python run.py train
```

Useful flags:
- `--force-target-span` — retrain target span model even if checkpoint exists
- `--force-preprocess` — regenerate embeddings/features
- `--allow-stance-fail` — continue even if stance polarity checks fail
- `--download-stance-data` — download external CONAN stance data before training

If you want to run stages manually:

```bash
python backend/rl_training/train_target_span_model.py
python backend/data/preprocess.py
python backend/rl_training/train.py
```

### 4. Run Backend API

```bash
# Start FastAPI server
python backend/api/app.py

# Or:
python run.py serve --backend-only
```

Backend will be available at `http://localhost:8000`

### 5. Run Frontend

```bash
# Install dependencies
cd frontend
npm install

# Start development server
npm run dev
```

Or start backend + frontend together:

```bash
python run.py serve
```

Frontend will be available at `http://localhost:5173`

## Usage

1. Open `http://localhost:5173` in your browser
2. Type a comment in the text area
3. Click "Moderate" to get AI decision
4. View:
   - **Decision**: Action taken (keep/warn/remove/temp_ban/perma_ban)
   - **Confidence**: Model certainty
   - **Reasoning**: Natural language explanation
   - **Toxicity Analysis**: Breakdown by category
   - **Alternative Actions**: Q-values for all actions
5. Use the feedback buttons to rate decisions (Too lenient/Just right/Too harsh).
   Feedback is logged to `backend/data/feedback.jsonl` and can trigger online updates.

## Architecture Details

### RL Environment

**State Space** (772 dimensions):
- Comment embedding: 768 dims (DistilBERT CLS token)
- Target features: 4 dims (target_presence, hate_prob, offensive_prob, normal_prob)

**Action Space** (5 discrete actions):
- 0: Keep
- 1: Warn
- 2: Remove
- 3: Temporary Ban
- 4: Permanent Ban

**Reward Function**:
```python
diff = toxicity_score - action_severity
reward = 1.0 - 2.0 * diff²
if diff < 0:  # over-moderation penalty
    reward -= 0.5 * |diff|
```

When target features are available, the environment uses target-aware toxicity
(`target_toxicity.npy`) for reward alignment.

**Sampling**:
- Toxic examples are oversampled during training (default `toxic_sample_prob=0.35`)

### Policy Network

```
Input (772) -> Comment Processor (768->256)
            -> Context Processor (4->64)
            -> Q-Network (320->256->5)
```

Returns Q-values for each action.

### Training

- **Algorithm**: Double DQN
- **Experience Replay**: 100K capacity buffer
- **Target Network**: Soft updates (tau=0.005)
- **Exploration**: epsilon-greedy (1.0 -> 0.05)
- **Optimizer**: Adam (lr=1e-4)
- **Batch Size**: 128
- **Episodes**: 1000
- **Steps per episode**: 500

## API Endpoints

### `POST /api/moderate`

Moderate a comment.

**Request:**
```json
{
  "comment": "Your comment here"
}
```

**Response:**
```json
{
  "decision": "warn",
  "confidence": 0.87,
  "reasoning": "Moderate insult detected (0.68). Borderline content - user warned. High confidence decision.",
  "toxicity_breakdown": {
    "toxicity": 0.65,
    "severe_toxicity": 0.12,
    "obscene": 0.34,
    "threat": 0.08,
    "insult": 0.68,
    "identity_attack": 0.05
  },
  "alternative_actions": [
    {"action": "warn", "q_value": 2.34, "probability": 0.87},
    {"action": "remove", "q_value": 1.12, "probability": 0.09},
    {"action": "keep", "q_value": 0.45, "probability": 0.03}
  ],
  "q_values": [0.45, 2.34, 1.12, -0.23, -1.45]
}
```

### `POST /api/feedback`

Submit feedback for a decision and optionally update the policy online.

**Request:**
```json
{
  "comment": "Your comment here",
  "decision": "warn",
  "feedback": "good"
}
```

Valid feedback values: `good`, `bad`, `too_harsh`, `too_soft`.

**Response:**
```json
{
  "status": "ok",
  "loss": 0.0123,
  "updated": true,
  "update_steps": 10,
  "batch_size": 32,
  "buffer_size": 128,
  "saved": false
}
```

### `GET /api/metrics`

Get training statistics.

### `GET /api/examples`

Get sample moderation examples.

## Performance Notes

- **Data preprocessing**: ~30 minutes (600K comments)
- **Training**: 2-4 hours on CPU (1000 episodes)
- **Inference**: ~100ms per comment
- **GPU support**: Uncomment torch-cuda in requirements.txt

## Datasets Used

- **Jigsaw/Conversation AI**: Toxic comment datasets
- **HateXplain**: 20k posts with target annotations, rationale spans, and 3-class labels
- **Ethos, SBIC, Measuring Hate Speech, HateCheck**: Optional supplementary datasets
- **Hugging Face Transformers**: DistilBERT encoder
- **Detoxify**: Toxicity classification
- **Gymnasium**: RL environment framework

---

**Built with PyTorch, FastAPI, and React**
