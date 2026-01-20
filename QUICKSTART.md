# Quick Start Guide

Get up and running in 5 steps:

## Step 1: Install Dependencies

```bash
pip install -r backend/requirements.txt
cd frontend
npm install
cd ..
```

## Step 2: Provide Training Data

Place a CSV at `backend/data/train.csv` with a `comment_text` column and toxicity label columns
(`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).

Preprocessing uses the base DistilBERT encoder (no fine-tuning).

Optional: train a hate/offensive/neither head using `backend/data/archive/labeled_data.csv`:

```bash
python backend/rl_training/train_hate_speech_head.py
```

If the head exists, preprocessing will also generate `backend/data/hate_scores.npy`.

## Step 3: Preprocess Data

```bash
python backend/data/preprocess.py
```

## Step 4: Train the Model

```bash
# All-in-one training (hate head + embeddings + DQN)
python backend/rl_training/train_all.py

# Or train only the DQN agent
# python backend/rl_training/train.py

# Watch training progress:
# Episode   10 | Reward:   45.23 | Loss: 0.1234 | Iae: 0.950
# Episode   20 | Reward:   67.89 | Loss: 0.0987 | Iae: 0.903
# ...
```

## Step 5: Run the Application

**Terminal 1 - Backend:**
```bash
python backend/api/app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Open browser**: http://localhost:5173

## Test It Out

Try these comments:

1. "This is a great article!" → **Keep**
2. "You're an idiot" → **Warn**
3. "I hate you all" → **Remove**
4. "Kill yourself" → **Temp Ban**

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'torch'`
**Fix**: Run `pip install -r backend/requirements.txt`

### Issue: `Error: backend/data/train.csv not found`
**Fix**: Place your dataset at `backend/data/train.csv` with required columns, then rerun preprocessing

### Issue: Frontend shows "Network Error"
**Fix**: Ensure backend is running on port 8000

### Issue: "No trained model found"
**Fix**: Complete Step 4 (training) first

## What's Next?

- Read [README.md](README.md) for full documentation
- Explore the training metrics
- Modify the reward function
- Add new features (dashboard, feedback system, etc.)

## System Requirements

- **Python**: 3.8+
- **Node.js**: 16+
- **RAM**: 8GB minimum (16GB recommended)
- **Disk**: 5GB free space
- **Time**: ~3-5 hours total (mostly automated)

---

**Need help?** Check the [README.md](README.md) or open an issue.
