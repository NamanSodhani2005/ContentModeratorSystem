"""
FastAPI backend for content moderation system.
"""

import os # File operations
import sys # System operations
import json # JSON logging
from datetime import datetime # Timestamping
from pathlib import Path # Path utilities
import torch # PyTorch framework
import numpy as np # Array operations
from fastapi import FastAPI, HTTPException # FastAPI framework
from fastapi.middleware.cors import CORSMiddleware # CORS middleware
from pydantic import BaseModel # Request models
from transformers import DistilBertTokenizer, DistilBertModel # BERT models
from detoxify import Detoxify # Toxicity detection

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent)) # Add parent to path

from rl_training.models.policy_network import PolicyNetwork # Q-network
from rl_training.agents.dqn_agent import DQNAgent # DQN agent

# Initialize FastAPI app
app = FastAPI(title="Content Moderation API", version="1.0.0") # Create app

# Add CORS middleware
app.add_middleware( # Add CORS
    CORSMiddleware, # CORS middleware
    allow_origins=["*"], # Allow all origins
    allow_credentials=True, # Allow credentials
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"], # Allow all headers
)

# Global models (loaded on startup)
agent = None # DQN agent
tokenizer = None # BERT tokenizer
embedder = None # BERT embedder
detoxify_model = None # Toxicity model
embedding_device = None # Device for embeddings
feedback_count = 0 # Feedback counter

# Action mappings
ACTION_NAMES = { # Action ID to name
    0: "keep", # Keep content
    1: "warn", # Issue warning
    2: "remove", # Remove content
    3: "temp_ban", # Temporary ban
    4: "perma_ban" # Permanent ban
}

ACTION_IDS = {v: k for k, v in ACTION_NAMES.items()} # Action name to ID

ACTION_COLORS = { # Action colors
    "keep": "green", # Green for keep
    "warn": "yellow", # Yellow for warn
    "remove": "orange", # Orange for remove
    "temp_ban": "red", # Red for temp ban
    "perma_ban": "darkred" # Dark red for perma ban
}

FEEDBACK_REWARDS = { # Feedback reward mapping
    "good": 1.0,
    "bad": -1.0,
    "too_harsh": -1.0,
    "too_soft": -1.0
}

# Request/Response models
class ModerationRequest(BaseModel): # Request model
    comment: str # Comment text

class ModerationResponse(BaseModel): # Response model
    decision: str # Moderation decision
    confidence: float # Decision confidence
    reasoning: str # Explanation
    toxicity_breakdown: dict # Toxicity scores
    alternative_actions: list # Other possible actions
    q_values: list # Q-values

class FeedbackRequest(BaseModel): # Feedback request model
    comment: str # Comment text
    decision: str # Action taken
    feedback: str # good | bad | too_harsh | too_soft

@app.on_event("startup") # Startup event
async def load_models(): # Load models
    """Load all models on startup."""
    global agent, tokenizer, embedder, detoxify_model, embedding_device # Global models

    print("Loading models...")

    # Load DistilBERT for embeddings
    print("  Loading DistilBERT...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Select device
    embedding_device = device # Store device
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased') # Load tokenizer
    embedder = DistilBertModel.from_pretrained('distilbert-base-uncased') # Load embedder
    embedder.to(device) # Move to device
    embedder.eval() # Set eval mode

    # Load Detoxify for toxicity analysis
    print("  Loading Detoxify...")
    detoxify_model = Detoxify('original') # Load detoxify

    # Load trained DQN agent
    print("  Loading DQN agent...")
    agent_device = device.type # DQN device
    policy_network = PolicyNetwork() # Create policy network
    target_network = PolicyNetwork() # Create target network

    agent = DQNAgent( # Create agent
        policy_network=policy_network, # Policy network
        target_network=target_network, # Target network
        device=agent_device # Device
    )

    # Load checkpoint if available
    model_path = Path('backend/saved_models/dqn_final.pt') # Model path
    if model_path.exists(): # Check if exists
        agent.load(model_path) # Load model
        print(f"  ✓ Loaded trained model from {model_path}")
    else: # Model not found
        print(f"  ⚠ No trained model found. Using untrained agent.")
        print(f"    Train model first: python backend/rl_training/train.py")

    print("✓ All models loaded successfully!")

def embed_comment(comment_text): # Embed comment
    """Generate 768-dim embedding for comment using DistilBERT."""
    inputs = tokenizer( # Tokenize text
        comment_text, # Comment text
        return_tensors='pt', # Return tensors
        truncation=True, # Truncate if long
        max_length=128, # Max 128 tokens
        padding='max_length' # Pad to max
    ).to(embedding_device) # Move to device

    with torch.no_grad(): # No gradients
        outputs = embedder(**inputs) # Forward pass
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy() # Get CLS token

    return embedding # Return embedding

def construct_state(comment_embedding): # Build state vector
    """
    Construct state vector from comment embedding.
    [comment_embedding(768), user_history(10), platform_metrics(5)]
    """
    # Mock user history (in production, fetch from database)
    user_history = np.array([ # User history
        0.2, # avg_toxicity
        0.0, # warnings_received
        0.0, # removals
        0.0, # temp_bans
        0.0, # perma_bans
        1.0, # activity_level
        0.8, # engagement_score
        0.0, # appeal_count
        10.0, # days_active
        5.0 # posts_count
    ])

    # Mock platform metrics
    platform_metrics = np.array([ # Platform metrics
        0.85, # platform_health
        0.80, # user_satisfaction
        0.05, # false_positive_rate
        0.15, # moderation_rate
        0.5 # time_step_norm
    ])

    state = np.concatenate([comment_embedding, user_history, platform_metrics]).astype(np.float32) # Concatenate all
    return state # Return state

def generate_explanation(comment, action, q_values, toxicity_scores, override_reason=None): # Generate explanation
    """Generate natural language explanation for moderation decision."""
    action_name = ACTION_NAMES[action] # Get action name
    reasoning_parts = [] # Reasoning list

    # Analyze toxicity
    if not toxicity_scores: # No toxicity scores available
        max_toxicity_type = "toxicity"
        max_toxicity_score = 0.0
    else:
        max_toxicity_type = max(toxicity_scores, key=toxicity_scores.get) # Find highest toxicity
        max_toxicity_score = toxicity_scores[max_toxicity_type] # Get score

    if max_toxicity_score > 0.7: # High toxicity
        reasoning_parts.append(f"High {max_toxicity_type} score ({max_toxicity_score:.2f})") # Add reasoning
    elif max_toxicity_score > 0.5: # Moderate toxicity
        reasoning_parts.append(f"Moderate {max_toxicity_type} detected ({max_toxicity_score:.2f})") # Add reasoning
    else: # Low toxicity
        reasoning_parts.append(f"Low toxicity detected ({max_toxicity_score:.2f})") # Add reasoning

    # Action-specific reasoning
    if action_name == "keep": # Keep action
        reasoning_parts.append("Content appears acceptable") # Add reasoning
    elif action_name == "warn": # Warn action
        reasoning_parts.append("Borderline content - user warned") # Add reasoning
    elif action_name == "remove": # Remove action
        reasoning_parts.append("Content violates community guidelines") # Add reasoning
    elif action_name == "temp_ban": # Temp ban
        reasoning_parts.append("Severe violation - temporary suspension") # Add reasoning
    elif action_name == "perma_ban": # Perma ban
        reasoning_parts.append("Extreme violation - permanent ban") # Add reasoning

    if override_reason: # Rule-based override explanation
        reasoning_parts.append(override_reason) # Add override reason

    if not override_reason: # Only use Q-values when no override
        # Q-value confidence
        q_diff = q_values[action] - np.sort(q_values)[-2] # Q-value margin
        if q_diff > 1.0: # High confidence
            reasoning_parts.append("High confidence decision") # Add reasoning
        elif q_diff > 0.5: # Moderate confidence
            reasoning_parts.append("Moderate confidence") # Add reasoning
        else: # Low confidence
            reasoning_parts.append("Low confidence - borderline case") # Add reasoning

    return ". ".join(reasoning_parts) + "." # Join and return

def log_feedback_entry(entry): # Log feedback to file
    """Append feedback entry to JSONL log."""
    log_path = Path('backend/data/feedback.jsonl') # Feedback log path
    log_path.parent.mkdir(exist_ok=True) # Ensure directory
    with open(log_path, 'a', encoding='utf-8') as handle: # Append mode
        handle.write(json.dumps(entry) + "\n") # Write entry

@app.get("/") # Root endpoint
async def root(): # Health check
    """Health check endpoint."""
    return { # Return status
        "status": "online", # Status
        "message": "Content Moderation API", # Message
        "version": "1.0.0" # Version
    }

@app.post("/api/moderate", response_model=ModerationResponse) # Moderation endpoint
async def moderate_comment(request: ModerationRequest): # Moderate comment
    """
    Moderate a comment and return decision with explanation.
    """
    try: # Try moderation
        comment = request.comment.strip() # Clean comment

        if not comment: # Empty comment check
            raise HTTPException(status_code=400, detail="Comment cannot be empty") # Error

        # 1. Embed comment
        comment_embedding = embed_comment(comment) # Get embedding

        # 2. Construct state
        state = construct_state(comment_embedding) # Build state vector

        # 3. Get agent decision
        action, q_values, attention_weights = agent.select_action(state, eval_mode=True) # Get action

        # 4. Get toxicity breakdown
        toxicity_scores = detoxify_model.predict(comment) if detoxify_model else {} # Get toxicity scores

        # 5. Generate explanation
        reasoning = generate_explanation(comment, action, q_values, toxicity_scores, None) # Generate reasoning

        # 6. Calculate confidence (softmax over Q-values)
        q_exp = np.exp(q_values - np.max(q_values)) # Exp Q-values
        q_probs = q_exp / q_exp.sum() # Softmax probabilities
        confidence = float(q_probs[action]) # Confidence score

        # 7. Get alternative actions
        action_indices = np.argsort(q_values)[::-1] # Sort actions by Q
        alternative_actions = [ # Build alternatives list
            {
                "action": ACTION_NAMES[int(idx)], # Action name
                "q_value": float(q_values[int(idx)]), # Q-value
                "probability": float(q_probs[int(idx)]) # Probability
            }
            for idx in action_indices # For each action
        ]

        return ModerationResponse( # Return response
            decision=ACTION_NAMES[action], # Decision name
            confidence=confidence, # Confidence score
            reasoning=reasoning, # Explanation
            toxicity_breakdown={k: float(v) for k, v in toxicity_scores.items()}, # Toxicity scores
            alternative_actions=alternative_actions, # Alternative actions
            q_values=q_values.tolist() # Q-values list
        )

    except Exception as e: # Handle errors
        raise HTTPException(status_code=500, detail=f"Moderation failed: {str(e)}") # Return error

@app.post("/api/feedback") # Feedback endpoint
async def submit_feedback(request: FeedbackRequest): # Handle feedback
    """
    Accept user feedback and update the DQN agent online.
    """
    global feedback_count # Feedback counter

    try: # Try feedback update
        if agent is None: # Agent missing
            raise HTTPException(status_code=503, detail="Agent not loaded") # Error

        comment = request.comment.strip() # Clean comment
        decision = request.decision.strip().lower() # Normalize decision
        feedback = request.feedback.strip().lower() # Normalize feedback

        if not comment: # Empty comment
            raise HTTPException(status_code=400, detail="Comment cannot be empty") # Error
        if decision not in ACTION_IDS: # Invalid decision
            raise HTTPException(status_code=400, detail="Invalid decision") # Error
        if feedback not in FEEDBACK_REWARDS: # Invalid feedback
            raise HTTPException(status_code=400, detail="Invalid feedback") # Error

        # Build state for feedback update
        comment_embedding = embed_comment(comment) # Embed comment
        state = construct_state(comment_embedding) # Build state vector

        # Reward based on feedback
        action = ACTION_IDS[decision] # Action ID
        reward = FEEDBACK_REWARDS[feedback] # Reward signal
        next_state = state # Terminal transition
        done = True # Terminal step

        agent.replay_buffer.push(state, action, reward, next_state, done) # Store transition
        loss = agent.train_step(batch_size=1) # Online update

        # Soft hint for harsh/soft feedback (adjacent action)
        if feedback in ["too_harsh", "too_soft"]: # Directional feedback
            if feedback == "too_harsh": # Suggest softer action
                suggested_action = max(0, action - 1) # One step softer
            else: # too_soft
                suggested_action = min(4, action + 1) # One step harsher
            if suggested_action != action: # Only if different
                agent.replay_buffer.push(state, suggested_action, 0.5, next_state, done) # Bonus signal
                agent.train_step(batch_size=1) # Train on hint

        feedback_count += 1 # Increment counter

        if feedback_count % 20 == 0: # Periodic save
            save_path = Path('backend/saved_models/dqn_final.pt') # Save path
            save_path.parent.mkdir(exist_ok=True) # Ensure directory
            agent.save(save_path) # Save updated model

        log_feedback_entry({ # Log feedback
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "comment": comment,
            "decision": decision,
            "feedback": feedback,
            "reward": reward
        })

        loss_value = float(loss) if loss is not None else None # Normalize loss
        return {"status": "ok", "loss": loss_value} # Response

    except HTTPException: # Pass through HTTP errors
        raise
    except Exception as e: # Handle errors
        raise HTTPException(status_code=500, detail=f"Feedback failed: {str(e)}") # Return error

@app.get("/api/metrics") # Metrics endpoint
async def get_metrics(): # Get training metrics
    """Get training metrics if available."""
    stats_path = Path('backend/saved_models/training_stats.npz') # Stats path

    if not stats_path.exists(): # Check if exists
        return {"message": "No training stats available"} # Return message

    stats = np.load(stats_path) # Load stats

    return { # Return metrics
        "total_episodes": len(stats['episode_rewards']), # Total episodes
        "avg_reward": float(np.mean(stats['episode_rewards'][-100:])), # Average reward
        "final_epsilon": float(stats['epsilon_values'][-1]), # Final epsilon
        "final_platform_health": float(stats['platform_health'][-1]), # Final health
        "final_false_positive_rate": float(stats['false_positive_rates'][-1]) # Final FP rate
    }

@app.get("/api/examples") # Examples endpoint
async def get_examples(): # Get example comments
    """Return sample moderation examples."""
    examples = [ # Example list
        {"comment": "This is a great article!", "expected": "keep"}, # Positive comment
        {"comment": "You're an idiot", "expected": "warn"}, # Mild insult
        {"comment": "I hate you all", "expected": "remove"}, # Hateful comment
        {"comment": "Kill yourself", "expected": "temp_ban"}, # Severe threat
        {"comment": "Threatening violence against [group]", "expected": "perma_ban"} # Extreme threat
    ]
    return examples # Return examples

if __name__ == "__main__":
    import uvicorn # Uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000) # Run server
