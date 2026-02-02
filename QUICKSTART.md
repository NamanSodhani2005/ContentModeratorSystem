# Quick Start Guide

Get up and running in 5 steps:

## Step 1: Install Dependencies

```bash
pip install -r backend/requirements.txt
cd frontend
npm install
cd ..
```

Frontend styling is pure CSS (no Tailwind or PostCSS setup required).

## Step 2: Provide Training Data

Place a CSV at `backend/data/train.csv` with a `comment_text` column and toxicity label columns
(`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).

Optional datasets for extra heads/target features:
- `backend/data/archive/labeled_data.csv` (hate/offensive head)
- `backend/data/dataset.json` (HateXplain)
- `backend/data/Ethos_Dataset_Binary.csv`, `backend/data/Ethos_Dataset_Multi_Label.csv`
- `backend/data/SBIC.v2.agg.trn.csv`
- `backend/data/measuring_hate_speech.csv`
- `backend/data/hate-speech-and-offensive-language.csv`

Optional stance augmentation:

```bash
python backend/data/augment_stance.py
```

## Step 3: Train the Models

Recommended one-shot pipeline:

```bash
python run.py train
```

Training uses balanced sampling and reward shaping by default to reduce toxic under-moderation.

Or run the full pipeline directly:

```bash
python backend/rl_training/train_all.py
```

## Step 4: Run the Application

One command (backend + frontend):

```bash
python run.py serve
```

Or run separately:

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

## Step 5: Test It Out

Try these comments:

1. "This is a great article!" -> **Keep**
2. "You're an idiot" -> **Warn**
3. "I hate you all" -> **Remove**
4. "Kill yourself" -> **Temp Ban**

Use the feedback buttons to rate decisions (Too lenient/Just right/Too harsh).
Feedback is logged to `backend/data/feedback.jsonl`.

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'torch'`
**Fix**: Run `pip install -r backend/requirements.txt`

### Issue: `Error: backend/data/train.csv not found`
**Fix**: Place your dataset at `backend/data/train.csv` with required columns, then rerun preprocessing

### Issue: Frontend shows "Network Error"
**Fix**: Ensure backend is running on port 8000

### Issue: "No trained model found"
**Fix**: Complete Step 3 (training) first

## What's Next?

- Read [README.md](README.md) for full documentation
- Explore the training metrics
- Modify the reward function
- Add new features (dashboard, feedback analytics, etc.)

## System Requirements

- **Python**: 3.8+
- **Node.js**: 16+
- **RAM**: 8GB minimum (16GB recommended)
- **Disk**: 5GB free space
- **Time**: ~3-5 hours total (mostly automated)

---
