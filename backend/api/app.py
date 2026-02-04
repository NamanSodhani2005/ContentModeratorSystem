"""
FastAPI backend for content moderation system.
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertModel
from detoxify import Detoxify

sys.path.append(str(Path(__file__).parent.parent))

from rl_training.models.policy_network import PolicyNetwork
from rl_training.agents.dqn_agent import DQNAgent
from rl_training.models.hate_speech_head import HateSpeechHead
from rl_training.models.target_span_model import TargetSpanToxicityModel

app = FastAPI(title="Content Moderation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Runtime state
agent = None
tokenizer = None
embedder = None
detoxify_model = None
hate_speech_head = None
target_span_model = None
embedding_device = None
feedback_count = 0

ACTION_NAMES = {0: "keep", 1: "warn", 2: "remove", 3: "temp_ban", 4: "perma_ban"}
ACTION_IDS = {v: k for k, v in ACTION_NAMES.items()}
ACTION_COLORS = {"keep": "green", "warn": "yellow", "remove": "orange", "temp_ban": "red", "perma_ban": "darkred"}
FEEDBACK_REWARDS = {"good": 1.0, "bad": -1.0, "too_harsh": -1.0, "too_soft": -1.0}

# Feedback tuning
FEEDBACK_MIN_BATCH = 8
FEEDBACK_BATCH_SIZE = 32
FEEDBACK_TRAIN_STEPS = 10
FEEDBACK_TARGET_TAU = 0.01
FEEDBACK_SAVE_EVERY = 20

class ModerationRequest(BaseModel):
    comment: str

class ModerationResponse(BaseModel):
    decision: str
    confidence: float
    reasoning: str
    toxicity_breakdown: dict
    alternative_actions: list
    q_values: list

class FeedbackRequest(BaseModel):
    """Feedback payload (good | bad | too_harsh | too_soft)."""

    comment: str
    decision: str
    feedback: str

@app.on_event("startup")
async def load_models():
    """Load all models on startup."""
    global agent, tokenizer, embedder, detoxify_model, embedding_device, hate_speech_head, target_span_model

    print("Loading models...")

    print("  Loading DistilBERT...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding_device = device
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    embedder = DistilBertModel.from_pretrained('distilbert-base-uncased')
    embedder.to(device)
    embedder.eval()

    hate_head_path = Path('backend/saved_models/hate_speech_head.pt')
    if hate_head_path.exists():
        hate_speech_head = HateSpeechHead()
        hate_speech_head.load_state_dict(torch.load(hate_head_path, map_location=device))
        hate_speech_head.to(device)
        hate_speech_head.eval()
        print("  [OK] Loaded hate speech head")

    target_span_path = Path('backend/saved_models/target_span_model.pt')
    if target_span_path.exists():
        target_span_model = TargetSpanToxicityModel(num_tox_classes=3)
        target_span_model.load_state_dict(torch.load(target_span_path, map_location=device))
        target_span_model.to(device)
        target_span_model.eval()
        print("  [OK] Loaded target span model")

    print("  Loading Detoxify...")
    detoxify_model = Detoxify('original')

    print("  Loading DQN agent...")
    agent_device = device.type
    policy_network = PolicyNetwork(context_dim=22)
    target_network = PolicyNetwork(context_dim=22)

    agent = DQNAgent(
        policy_network=policy_network,
        target_network=target_network,
        device=agent_device
    )

    model_path = Path('backend/saved_models/dqn_final.pt')
    if model_path.exists():
        agent.load(model_path)
        print(f"  [OK] Loaded trained model from {model_path}")
    else:
        print("  [WARN] No trained model found. Using untrained agent.")
        print("    Train model first: python backend/rl_training/train.py")

    agent.policy_network.eval()
    agent.target_network.eval()

    print("[OK] All models loaded successfully!")


def embed_comment(comment_text):
    """Generate 768-dim embedding for comment using DistilBERT."""
    inputs = tokenizer(
        comment_text,
        return_tensors='pt',
        truncation=True,
        max_length=128,
        padding='max_length'
    ).to(embedding_device)

    with torch.no_grad():
        outputs = embedder(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

    return embedding


def compute_target_features(comment_text):
    """
    Compute target detection features using TargetSpanToxicityModel.
    Returns: [target_presence, hate_prob, offensive_prob, normal_prob] (4-dim)
    """
    if target_span_model is None:
        return np.zeros(4, dtype=np.float32)

    inputs = tokenizer(
        comment_text,
        return_tensors='pt',
        truncation=True,
        max_length=128,
        padding='max_length'
    ).to(embedding_device)

    with torch.no_grad():
        outputs = target_span_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

        token_probs = torch.softmax(outputs['token_logits'], dim=-1)[..., 1]
        token_probs = token_probs * inputs['attention_mask']
        tox_probs = torch.softmax(outputs['toxicity_logits'], dim=-1)[0]

        target_presence = token_probs.max().item()
        if tox_probs.numel() == 3:
            hate_prob = tox_probs[0].item()
            offensive_prob = tox_probs[1].item()
            normal_prob = tox_probs[2].item()
        else:
            toxic_prob = tox_probs[1].item() if tox_probs.numel() > 1 else 0.0
            hate_prob = 0.0
            offensive_prob = toxic_prob
            normal_prob = 1.0 - toxic_prob

    return np.array([target_presence, hate_prob, offensive_prob, normal_prob], dtype=np.float32)


def construct_state(comment_embedding, comment_text):
    """
    Construct state vector from comment embedding and text.
    [comment_embedding(768), hate_scores(3), target_features(4), user_history(10), platform_metrics(5)]
    """
    # user_history order: avg_toxicity, warnings_received, removals, temp_bans, perma_bans,
    # activity_level, engagement_score, appeal_count, days_active, posts_count
    user_history = np.array([
        0.2,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.8,
        0.0,
        10.0,
        5.0
    ])

    # platform_metrics order: platform_health, user_satisfaction, false_positive_rate, moderation_rate, time_step_norm
    platform_metrics = np.array([
        0.85,
        0.80,
        0.05,
        0.15,
        0.5
    ])

    hate_scores = np.zeros(3, dtype=np.float32)
    if hate_speech_head is not None:
        with torch.no_grad():
            embed_tensor = torch.tensor(comment_embedding, dtype=torch.float32, device=embedding_device).unsqueeze(0)
            logits = hate_speech_head(embed_tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            hate_scores = probs.astype(np.float32)

    target_features = compute_target_features(comment_text)

    state = np.concatenate([
        comment_embedding,
        hate_scores,
        target_features,
        user_history,
        platform_metrics
    ]).astype(np.float32)

    return state


def generate_explanation(comment, action, q_values, toxicity_scores, override_reason=None):
    """Generate natural language explanation for moderation decision."""
    action_name = ACTION_NAMES[action]
    reasoning_parts = []

    if not toxicity_scores:
        max_toxicity_type = "toxicity"
        max_toxicity_score = 0.0
    else:
        max_toxicity_type = max(toxicity_scores, key=toxicity_scores.get)
        max_toxicity_score = toxicity_scores[max_toxicity_type]

    if max_toxicity_score > 0.7:
        reasoning_parts.append(f"High {max_toxicity_type} score ({max_toxicity_score:.2f})")
    elif max_toxicity_score > 0.5:
        reasoning_parts.append(f"Moderate {max_toxicity_type} detected ({max_toxicity_score:.2f})")
    else:
        reasoning_parts.append(f"Low toxicity detected ({max_toxicity_score:.2f})")

    if action_name == "keep":
        reasoning_parts.append("Content appears acceptable")
    elif action_name == "warn":
        reasoning_parts.append("Borderline content - user warned")
    elif action_name == "remove":
        reasoning_parts.append("Content violates community guidelines")
    elif action_name == "temp_ban":
        reasoning_parts.append("Severe violation - temporary suspension")
    elif action_name == "perma_ban":
        reasoning_parts.append("Extreme violation - permanent ban")

    if override_reason:
        reasoning_parts.append(override_reason)

    if not override_reason:
        q_diff = q_values[action] - np.sort(q_values)[-2]
        if q_diff > 1.0:
            reasoning_parts.append("High confidence decision")
        elif q_diff > 0.5:
            reasoning_parts.append("Moderate confidence")
        else:
            reasoning_parts.append("Low confidence - borderline case")

    return ". ".join(reasoning_parts) + "."


def log_feedback_entry(entry):
    """Append feedback entry to JSONL log."""
    log_path = Path('backend/data/feedback.jsonl')
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as handle:
        handle.write(json.dumps(entry) + "\n")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "online", "message": "Content Moderation API", "version": "1.0.0"}


@app.post("/api/moderate", response_model=ModerationResponse)
async def moderate_comment(request: ModerationRequest):
    """Moderate a comment and return decision with explanation."""
    try:
        comment = request.comment.strip()

        if not comment:
            raise HTTPException(status_code=400, detail="Comment cannot be empty")

        comment_embedding = embed_comment(comment)
        state = construct_state(comment_embedding, comment)
        action, q_values, attention_weights = agent.select_action(state, eval_mode=True)
        toxicity_scores = detoxify_model.predict(comment) if detoxify_model else {}
        reasoning = generate_explanation(comment, action, q_values, toxicity_scores, None)

        q_exp = np.exp(q_values - np.max(q_values))
        q_probs = q_exp / q_exp.sum()
        confidence = float(q_probs[action])

        action_indices = np.argsort(q_values)[::-1]
        alternative_actions = [
            {
                "action": ACTION_NAMES[int(idx)],
                "q_value": float(q_values[int(idx)]),
                "probability": float(q_probs[int(idx)])
            }
            for idx in action_indices
        ]

        return ModerationResponse(
            decision=ACTION_NAMES[action],
            confidence=confidence,
            reasoning=reasoning,
            toxicity_breakdown={k: float(v) for k, v in toxicity_scores.items()},
            alternative_actions=alternative_actions,
            q_values=q_values.tolist()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Moderation failed: {str(e)}")


@app.post("/api/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Accept user feedback and update the DQN agent online."""
    global feedback_count

    try:
        if agent is None:
            raise HTTPException(status_code=503, detail="Agent not loaded")

        comment = request.comment.strip()
        decision = request.decision.strip().lower()
        feedback = request.feedback.strip().lower()

        if not comment:
            raise HTTPException(status_code=400, detail="Comment cannot be empty")
        if decision not in ACTION_IDS:
            raise HTTPException(status_code=400, detail="Invalid decision")
        if feedback not in FEEDBACK_REWARDS:
            raise HTTPException(status_code=400, detail="Invalid feedback")

        comment_embedding = embed_comment(comment)
        state = construct_state(comment_embedding, comment)

        action = ACTION_IDS[decision]
        primary_reward = FEEDBACK_REWARDS[feedback]
        next_state = state
        done = True

        transitions = []
        # Translate feedback into training transitions.
        if feedback == "good":
            transitions.append((action, 1.0))
        elif feedback == "bad":
            transitions.append((action, -1.0))
        elif feedback == "too_harsh":
            transitions.append((action, -1.0))
            suggested_action = max(0, action - 1)
            if suggested_action != action:
                transitions.append((suggested_action, 1.0))
        elif feedback == "too_soft":
            transitions.append((action, -1.0))
            suggested_action = min(4, action + 1)
            if suggested_action != action:
                transitions.append((suggested_action, 1.0))

        for action_id, reward in transitions:
            agent.replay_buffer.push(state, action_id, reward, next_state, done)

        buffer_size = len(agent.replay_buffer)
        batch_size = min(FEEDBACK_BATCH_SIZE, buffer_size)
        losses = []
        updated = False
        if buffer_size >= FEEDBACK_MIN_BATCH:
            agent.policy_network.train()
            for _ in range(FEEDBACK_TRAIN_STEPS):
                loss = agent.train_step(batch_size=batch_size)
                if loss is not None:
                    losses.append(loss)
            agent.update_target_network(tau=FEEDBACK_TARGET_TAU)
            agent.policy_network.eval()
            updated = len(losses) > 0

        feedback_count += 1

        saved = False
        if feedback_count % FEEDBACK_SAVE_EVERY == 0:
            save_path = Path('backend/saved_models/dqn_final.pt')
            save_path.parent.mkdir(exist_ok=True)
            agent.save(save_path)
            saved = True

        log_feedback_entry({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "comment": comment,
            "decision": decision,
            "feedback": feedback,
            "reward": primary_reward
        })

        loss_value = float(np.mean(losses)) if losses else None
        return {
            "status": "ok",
            "loss": loss_value,
            "updated": updated,
            "update_steps": len(losses),
            "batch_size": batch_size if updated else 0,
            "buffer_size": buffer_size,
            "saved": saved
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback failed: {str(e)}")


@app.get("/api/metrics")
async def get_metrics():
    """Get training metrics if available."""
    stats_path = Path('backend/saved_models/training_stats.npz')

    if not stats_path.exists():
        return {"message": "No training stats available"}

    stats = np.load(stats_path)

    return {
        "total_episodes": len(stats['episode_rewards']),
        "avg_reward": float(np.mean(stats['episode_rewards'][-100:])),
        "final_epsilon": float(stats['epsilon_values'][-1]),
        "final_platform_health": float(stats['platform_health'][-1]),
        "final_false_positive_rate": float(stats['false_positive_rates'][-1])
    }


@app.get("/api/examples")
async def get_examples():
    """Return sample moderation examples."""
    return [
        {"comment": "This is a great article!", "expected": "keep"},
        {"comment": "You're an idiot", "expected": "warn"},
        {"comment": "I hate you all", "expected": "remove"},
        {"comment": "Kill yourself", "expected": "temp_ban"},
        {"comment": "Threatening violence against [group]", "expected": "perma_ban"}
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

