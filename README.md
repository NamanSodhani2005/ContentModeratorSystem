# Content Moderation RL System

An end-to-end content moderation stack that applies a DQN policy over DistilBERT embeddings with toxicity/target-span features, served via FastAPI and surfaced in a React UI. The system outputs discrete moderation actions (keep/warn/remove/temp_ban/perma_ban) with confidence, reasoning, and Q-value alternatives.

## Features

- **Deep Q-Network (DQN)** agent with attention-based policy network
- **Target-aware features** from a target span toxicity model (target presence + hate/offensive/normal probs)
- **Multi-objective reward function** balancing toxicity reduction, false positives, and user retention
- **Balanced sampling + reward shaping** to reduce toxic under-moderation
- **Explainable AI**: attention weights, Q-value transparency, and natural language reasoning
- **Real-time moderation** via FastAPI backend
- **Interactive React UI** with toxicity breakdown, alternative actions, and feedback loop
- **Online feedback updates** logged to JSONL for incremental policy tuning
- **Comprehensive training pipeline** with optional heads and data augmentation

## Tech Stack

### Backend
- **RL Framework**: PyTorch + Gymnasium
- **API**: FastAPI + Uvicorn
- **NLP**: Transformers (DistilBERT), Detoxify
- **Target span model**: DistilBERT + weak supervision
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
|   |   |-- augment_stance.py         # Optional stance augmentation
|   |   |-- preprocess.py             # Embeddings + feature generation
|   |   |-- target_lexicon.json       # Generated lexicon (optional)
|   |   `-- feedback.jsonl            # Runtime feedback log
|   |-- rl_training/
|   |   |-- environment/
|   |   |   `-- forum_env.py          # Gymnasium environment
|   |   |-- agents/
|   |   |   `-- dqn_agent.py          # DQN agent + replay buffer
|   |   |-- models/
|   |   |   |-- policy_network.py     # Attention-based Q-network
|   |   |   `-- target_span_model.py  # Target span toxicity model
|   |   |-- train.py                  # DQN training loop
|   |   |-- train_all.py              # Full training pipeline
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
|-- run.py
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

Optional datasets for extra heads/target features (train_target_span_model.py will use any it finds):
- `backend/data/archive/labeled_data.csv` (hate/offensive head)
- `backend/data/dataset.json` (HateXplain)
- `backend/data/Ethos_Dataset_Binary.csv`, `backend/data/Ethos_Dataset_Multi_Label.csv`
- `backend/data/SBIC.v2.agg.trn.csv`
- `backend/data/measuring_hate_speech.csv`
- `backend/data/hate-speech-and-offensive-language.csv`
- `backend/data/en_dataset_with_stop_words.csv`
- `backend/data/archive2/test (1).csv` (HateCheck)
- `backend/data/jigsaw-toxic-comment-classification-challenge/train.csv`

Optional stance augmentation (adds anti-hate/anti-abuse statements):

```bash
python backend/data/augment_stance.py
```

This writes `backend/data/stance_train.csv` and updates `backend/data/train.csv`
(backup: `backend/data/train.csv.stance.bak`).

### 2. Backend Setup

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Preprocess data and generate embeddings/features (~30 minutes)
python backend/data/preprocess.py
```

Preprocessing generates `embeddings.npy` and `labels.npy`, and will also generate
`hate_scores.npy`, `target_features.npy`, and `target_toxicity.npy` if the
corresponding heads are present. If you train a head after preprocessing,
rerun preprocessing to regenerate features.

### 3. Train the Model

Recommended one-shot pipeline (optional heads -> preprocess -> DQN):

```bash
python run.py train
```

Note: `run.py train` trains the target span model only when `backend/data/archive/labeled_data.csv`
is present. If you only have HateXplain or other target datasets, run
`python backend/rl_training/train_target_span_model.py` before preprocessing.

Or run the full pipeline directly:

```bash
python backend/rl_training/train_all.py
```

If you want to run stages manually:

```bash
python backend/rl_training/train_hate_speech_head.py
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

## Example Comments

Try these examples:

| Comment | Expected Decision |
|---------|------------------|
| "This is a great article!" | Keep |
| "You're an idiot" | Warn |
| "I hate you all" | Remove |
| "Kill yourself" | Temp Ban |
| "Threatening violence against [group]" | Perma Ban |

## Architecture Details

### RL Environment

**State Space** (790 dimensions):
- Comment embedding: 768 dims (DistilBERT)
- Hate/offensive scores: 3 dims
- Target features: 4 dims (target_presence, hate_prob, offensive_prob, normal_prob)
- User history: 10 dims (toxicity avg, warnings, bans, activity, etc.)
- Platform metrics: 5 dims (health, satisfaction, false positive rate, etc.)

**Action Space** (5 discrete actions):
- 0: Keep
- 1: Warn
- 2: Remove
- 3: Temporary Ban
- 4: Permanent Ban

**Reward Function**:
```python
reward = alignment
       - over_penalty
       - under_penalty
       + platform_health_bonus
```

When target features are available, the environment uses target-aware toxicity
(`target_toxicity.npy`) for reward alignment.

**Sampling**:
- Toxic examples are oversampled during training (default `toxic_sample_prob=0.5`)
- Toxicity threshold defaults to `0.5` when building toxic/non-toxic pools

### Policy Network

```
Input (790) -> Comment Processor (768->256)
            -> Context Processor (22->64)
            -> Attention Layer
            -> Q-Network -> Q-values (5)
```

- **Attention layer** provides interpretability
- Returns Q-values + attention weights

### Training

- **Algorithm**: Deep Q-Network (DQN)
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

## Research Angles

This project demonstrates several responsible AI concepts:

1. **Reward Misspecification**: Train unconstrained agent -> observe over-moderation
2. **Fairness Analysis**: Measure disparate impact across identity groups
3. **Interpretability**: Attention weights + Q-value transparency + natural language explanations
4. **Long-horizon Effects**: Simulate platform health over 1000+ steps

## Training Metrics

After training, you'll see:

```
Episode  990 | Reward:  125.34 | Loss: 0.0234 | Eps: 0.050 | Health: 0.89 | FP: 0.043

Final Statistics:
  Average reward (last 100): 123.45
  Average loss (last 100): 0.0198
  Final epsilon: 0.050
  Final platform health: 0.89
  Final false positive rate: 0.043
```

## Troubleshooting

### Backend won't start

**Error**: `No module named 'torch'`

**Solution**: Install dependencies
```bash
pip install -r backend/requirements.txt
```

### Data not found

**Error**: `Error: backend/data/train.csv not found`

**Solution**: Place your dataset at `backend/data/train.csv` with a `comment_text` column
and the toxicity label columns, then rerun preprocessing:
```bash
python backend/data/preprocess.py
```

### Frontend can't connect to backend

**Error**: `Network Error` in browser console

**Solution**: Ensure backend is running on port 8000
```bash
python backend/api/app.py
```

### Model not found warning

**Warning**: `No trained model found. Using untrained agent.`

**Solution**: Train the model first
```bash
python backend/rl_training/train.py
```

## Performance Notes

- **Data preprocessing**: ~30 minutes (50K comments)
- **Training**: 2-4 hours on CPU (1000 episodes)
- **Inference**: ~100ms per comment
- **GPU support**: Uncomment torch-cuda in requirements.txt

## Datasets Used

- **Jigsaw/Conversation AI**: Toxic comment datasets
- **Hugging Face**: Transformers library
- **Detoxify**: Toxicity classification models
- **Gymnasium**: RL environment framework
- **HateXplain, Ethos, SBIC, Measuring Hate Speech, HateCheck**: Optional datasets used for target-aware training

---

**Built with PyTorch, FastAPI, and React** 
