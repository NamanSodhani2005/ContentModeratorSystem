# Content Moderation RL System

A full-stack AI system demonstrating **responsible AI deployment** through Deep Reinforcement Learning for content moderation. Users type comments and receive moderation decisions (keep/warn/remove/temp_ban/perma_ban) with explanations.

## Features

- **Deep Q-Network (DQN)** agent with attention-based policy network
- **Multi-objective reward function** balancing toxicity reduction, false positives, and user retention
- **Explainable AI**: Attention weights, Q-value transparency, and natural language reasoning
- **Real-time moderation** via FastAPI backend
- **Interactive React UI** with toxicity breakdown and alternative actions
- **Comprehensive training pipeline** with experience replay and target networks

## Tech Stack

### Backend
- **RL Framework**: PyTorch + Gymnasium
- **API**: FastAPI + Uvicorn
- **NLP**: Transformers (DistilBERT), Detoxify
- **Data**: Pandas, NumPy

### Frontend
- **Framework**: React + Vite
- **Styling**: Tailwind CSS
- **HTTP**: Axios
- **Icons**: Lucide React

## Project Structure

```
content-moderation-rl/
├── backend/
│   ├── data/
│   │   ├── train.csv                # Your dataset (not tracked)
│   │   └── preprocess.py            # BERT embeddings generation
│   ├── rl_training/
│   │   ├── environment/
│   │   │   └── forum_env.py         # Gymnasium environment
│   │   ├── agents/
│   │   │   └── dqn_agent.py         # DQN agent + replay buffer
│   │   ├── models/
│   │   │   └── policy_network.py    # Attention-based Q-network
│   │   └── train.py                 # Training loop
│   │   └── train_all.py             # Full training pipeline
│   ├── api/
│   │   └── app.py                   # FastAPI server
│   ├── saved_models/                # Trained checkpoints
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── CommentInput.jsx
│   │   │   └── ModerationResult.jsx
│   │   ├── App.jsx
│   │   └── api.js
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- Dataset CSV at `backend/data/train.csv` (for training)

### 1. Prepare Data

Place a CSV at `backend/data/train.csv` with a `comment_text` column and toxicity label columns
(`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).

If you already have `embeddings.npy` and `labels.npy`, you can skip preprocessing.

Preprocessing uses the base DistilBERT encoder (no fine-tuning).

Optional: train a hate/offensive/neither head using `backend/data/archive/labeled_data.csv`:

```bash
python backend/rl_training/train_hate_speech_head.py
```

If the head exists, preprocessing will also generate `backend/data/hate_scores.npy` for RL training.

### 2. Backend Setup

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Preprocess data and generate embeddings (~30 minutes)
python backend/data/preprocess.py
```

### 3. Train the Model

All-in-one training (hate head + embeddings + DQN):

```bash
python backend/rl_training/train_all.py
```

Or train only the DQN agent (requires precomputed embeddings):

```bash
python backend/rl_training/train.py
```

### 4. Run Backend API

```bash
# Start FastAPI server
python backend/api/app.py

# Or with uvicorn directly:
uvicorn backend.api.app:app --reload --port 8000
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

Frontend will be available at `http://localhost:5173`

## Usage

1. Open `http://localhost:5173` in your browser
2. Type a comment in the text area
3. Click "Moderate" to get AI decision
4. View:
   - **Decision**: Action taken (keep/warn/remove/ban)
   - **Confidence**: Model certainty
   - **Reasoning**: Natural language explanation
   - **Toxicity Analysis**: Breakdown by category
   - **Alternative Actions**: Q-values for all actions

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

**State Space** (786 dimensions):
- Comment embedding: 768 dims (DistilBERT)
- Hate/offensive scores: 3 dims
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
reward = toxicity_reduction
       - false_positive_penalty
       + platform_health_bonus
       - harsh_action_penalty
```

### Policy Network

```
Input (786) → Comment Processor (768→256)
            → Context Processor (18→64)
            → Attention Layer
            → Q-Network → Q-values (5)
```

- **Attention layer** provides interpretability
- Returns Q-values + attention weights

### Training

- **Algorithm**: Deep Q-Network (DQN)
- **Experience Replay**: 100K capacity buffer
- **Target Network**: Soft updates (τ=0.005)
- **Exploration**: ε-greedy (1.0 → 0.05)
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

### `GET /api/metrics`

Get training statistics.

### `GET /api/examples`

Get sample moderation examples.

## Research Angles

This project demonstrates several responsible AI concepts:

1. **Reward Misspecification**: Train unconstrained agent → observe over-moderation
2. **Fairness Analysis**: Measure disparate impact across identity groups
3. **Interpretability**: Attention weights + Q-value transparency + natural language explanations
4. **Long-horizon Effects**: Simulate platform health over 1000+ steps

## Training Metrics

After training, you'll see:

```
Episode  990 | Reward:  125.34 | Loss: 0.0234 | ε: 0.050 | Health: 0.89 | FP: 0.043

Final Statistics:
  Average reward (last 100): 123.45
  Average loss (last 100): 0.0198
  Final epsilon: 0.050
  Final platform health: 0.89
  Final false positive rate: 0.043
```

## Future Extensions

- [ ] Dashboard with training metrics visualization (Recharts)
- [ ] Simulation panel showing long-term platform effects
- [ ] Examples gallery browsing past decisions
- [ ] Feedback system (user agrees/disagrees → RLHF)
- [ ] Constrained RL version (Lagrangian or CPO)
- [ ] Fairness auditing with demographic data
- [ ] Database for persistent decision storage

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

**Warning**: `⚠ No trained model found. Using untrained agent.`

**Solution**: Train the model first
```bash
python backend/rl_training/train.py
```

## Performance Notes

- **Data preprocessing**: ~30 minutes (50K comments)
- **Training**: 2-4 hours on CPU (1000 episodes)
- **Inference**: ~100ms per comment
- **GPU support**: Uncomment torch-cuda in requirements.txt

## License

MIT License

## Citation

If you use this project in your research, please cite:

```bibtex
@software{content_moderation_rl,
  title={Content Moderation RL System},
  author={Your Name},
  year={2024},
  description={Deep Reinforcement Learning for responsible content moderation}
}
```

## Acknowledgments

- **Jigsaw/Conversation AI**: Toxic comment datasets
- **Hugging Face**: Transformers library
- **Detoxify**: Toxicity classification models
- **OpenAI Gymnasium**: RL environment framework

---

**Built with PyTorch, FastAPI, and React** | Demonstrating responsible AI deployment
