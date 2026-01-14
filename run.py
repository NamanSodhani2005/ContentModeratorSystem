"""
Main runner for Content Moderation System.
Handles preprocessing, training, and API serving.
"""

import os # File operations
import sys # System operations
import numpy as np # Array operations
import pandas as pd # Data manipulation
import torch # PyTorch framework
from pathlib import Path # Path utilities
from tqdm import tqdm # Progress bars
from transformers import DistilBertTokenizer, DistilBertModel # BERT models

# Add backend to path
sys.path.append('backend') # Add backend to path

from rl_training.environment.forum_env import ForumEnvironment # RL environment
from rl_training.models.policy_network import PolicyNetwork # Q-network
from rl_training.agents.dqn_agent import DQNAgent # DQN agent
 

def preprocess_data(): # Preprocess data
    """Step 1: Generate embeddings from raw CSV"""
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)

    data_dir = Path('backend/data') # Data directory
    train_path = data_dir / 'train.csv' # CSV path

    if not train_path.exists(): # Check if exists
        print(f"❌ Error: {train_path} not found!")
        print("Please place train.csv in backend/data/")
        return False

    # Check if already processed
    if (data_dir / 'embeddings.npy').exists(): # Check if done
        print("✓ Embeddings already exist. Skipping preprocessing.")
        return True

    print("Loading CSV data...")
    df = pd.read_csv(train_path, nrows=50000) # Load 50K rows
    comments = df['comment_text'].fillna("").tolist() # Extract comments

    toxicity_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'] # Label columns
    labels = df[toxicity_cols].values # Extract labels

    print(f"✓ Loaded {len(comments)} comments")
    print("\nLoading DistilBERT model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Select device
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased') # Load tokenizer
    model = DistilBertModel.from_pretrained('distilbert-base-uncased') # Load model
    model.to(device) # Move to device
    model.eval() # Eval mode
    print(f"✓ Using device: {device}")

    print("\nGenerating embeddings (this takes ~30 minutes)...")
    batch_size = 32 # Batch size
    embeddings = [] # Store embeddings

    for idx in tqdm(range(0, len(comments), batch_size), desc="Progress"): # Process batches
        batch = comments[idx:idx + batch_size] # Get batch

        with torch.no_grad(): # No gradients
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device) # Tokenize
            outputs = model(**inputs) # Forward pass
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy() # Get CLS token
            embeddings.append(batch_embeddings) # Add to list

    embeddings = np.vstack(embeddings) # Stack all

    # Save files
    np.save(data_dir / 'embeddings.npy', embeddings) # Save embeddings
    np.save(data_dir / 'labels.npy', labels) # Save labels

    print(f"\n✓ Preprocessing complete!")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Toxicity rate: {labels[:, 0].mean():.2%}")

    return True

def train_model(): # Train DQN model
    """Step 2: Train DQN agent"""
    print("\n" + "="*60)
    print("STEP 2: TRAINING DQN AGENT")
    print("="*60)

    # Check if already trained
    if Path('backend/saved_models/dqn_final.pt').exists(): # Check if exists
        print("✓ Trained model already exists. Skipping training.")
        return True

    # Load data
    embeddings_path = 'backend/data/embeddings.npy' # Embeddings path
    labels_path = 'backend/data/labels.npy' # Labels path

    if not os.path.exists(embeddings_path): # Check if exists
        print("❌ Error: Run preprocessing first!")
        return False

    print("Loading data...")
    embeddings = np.load(embeddings_path) # Load embeddings
    labels = np.load(labels_path) # Load labels
    print(f"✓ Loaded {len(embeddings)} comments")

    # Initialize environment
    print("\nInitializing environment...")
    env = ForumEnvironment(embeddings, labels, max_steps=500) # Create env
    print(f"✓ State space: {env.observation_space.shape}")
    print(f"✓ Action space: {env.action_space.n}")

    # Initialize networks
    print("\nInitializing neural networks...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # Select device
    print(f"✓ Using device: {device}")

    policy_network = PolicyNetwork() # Policy network
    target_network = PolicyNetwork() # Target network
    print(f"✓ Parameters: {sum(p.numel() for p in policy_network.parameters()):,}")

    # Initialize agent
    print("\nInitializing DQN agent...")
    agent = DQNAgent( # Create agent
        policy_network=policy_network, # Policy network
        target_network=target_network, # Target network
        device=device, # Device
        lr=1e-4, # Learning rate
        gamma=0.99, # Discount factor
        epsilon_start=1.0, # Initial epsilon
        epsilon_end=0.05, # Final epsilon
        epsilon_decay=0.995 # Decay rate
    )
    print("✓ Agent ready")

    # Training loop
    print("\n" + "="*60)
    print("TRAINING (1000 episodes, ~2-4 hours)")
    print("="*60 + "\n")

    num_episodes = 1000 # Total episodes
    max_steps = 500 # Steps per episode
    batch_size = 128 # Batch size
    save_interval = 100 # Save every N episodes

    training_stats = { # Stats dictionary
        'episode_rewards': [], # Rewards
        'episode_lengths': [], # Lengths
        'losses': [], # Losses
        'epsilon_values': [], # Epsilon
        'platform_health': [], # Health
        'false_positive_rates': [] # FP rate
    }

    for episode in range(num_episodes): # Episode loop
        state, _ = env.reset() # Reset env
        episode_reward = 0 # Reset reward
        episode_loss = [] # Reset losses

        for step in range(max_steps): # Step loop
            action, q_values, attention = agent.select_action(state) # Select action
            next_state, reward, done, truncated, info = env.step(action) # Take step
            agent.replay_buffer.push(state, action, reward, next_state, done) # Store experience

            loss = agent.train_step(batch_size) # Train
            if loss is not None: # If trained
                episode_loss.append(loss) # Record loss

            episode_reward += reward # Accumulate reward
            state = next_state # Update state

            if done: # If done
                break # Exit loop

        # Update target network
        if episode % 10 == 0: # Every 10 episodes
            agent.update_target_network(tau=0.005) # Soft update

        agent.decay_epsilon() # Decay epsilon

        # Record stats
        training_stats['episode_rewards'].append(episode_reward) # Record reward
        training_stats['episode_lengths'].append(step + 1) # Record length
        training_stats['losses'].append(np.mean(episode_loss) if episode_loss else 0) # Record loss
        training_stats['epsilon_values'].append(agent.epsilon) # Record epsilon
        training_stats['platform_health'].append(info['platform_health']) # Record health
        training_stats['false_positive_rates'].append(info['false_positive_rate']) # Record FP rate

        # Print progress
        if episode % 10 == 0: # Every 10 episodes
            avg_reward = np.mean(training_stats['episode_rewards'][-10:]) # Avg reward
            avg_loss = np.mean(training_stats['losses'][-10:]) # Avg loss
            print(f"Episode {episode:4d}/{num_episodes} | " # Print progress
                  f"Reward: {avg_reward:7.2f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Health: {info['platform_health']:.2f} | "
                  f"FP: {info['false_positive_rate']:.3f}")

        # Save checkpoint
        if episode % save_interval == 0 and episode > 0: # Every N episodes
            save_dir = Path('backend/saved_models') # Save directory
            save_dir.mkdir(exist_ok=True) # Create directory
            save_path = save_dir / f'dqn_checkpoint_ep{episode}.pt' # Checkpoint path
            agent.save(save_path) # Save checkpoint

    # Save final model
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

    save_dir = Path('backend/saved_models') # Save directory
    save_dir.mkdir(exist_ok=True) # Create directory
    final_path = save_dir / 'dqn_final.pt' # Final path
    agent.save(final_path) # Save model

    # Save stats
    stats_path = save_dir / 'training_stats.npz' # Stats path
    np.savez(stats_path, **training_stats) # Save stats
    print(f"✓ Stats saved to {stats_path}")

    # Print final stats
    print(f"\nFinal Statistics:")
    print(f"  Avg reward (last 100): {np.mean(training_stats['episode_rewards'][-100:]):.2f}")
    print(f"  Avg loss (last 100): {np.mean(training_stats['losses'][-100:]):.4f}")
    print(f"  Final epsilon: {agent.epsilon:.3f}")
    print(f"  Final platform health: {training_stats['platform_health'][-1]:.2f}")
    print(f"  Final FP rate: {training_stats['false_positive_rates'][-1]:.3f}")

    return True

def run_api(): # Run API server
    """Step 3: Start API server"""
    print("\n" + "="*60)
    print("STEP 3: STARTING API SERVER")
    print("="*60)
    print("\nBackend will run on: http://localhost:8000")
    print("Frontend should run on: http://localhost:5173")
    print("\nPress Ctrl+C to stop\n")

    import uvicorn # Uvicorn server
    from backend.api.app import app # Import app

    uvicorn.run(app, host="0.0.0.0", port=8000) # Run server

def main(): # Main function
    """Run full pipeline"""
    print("\n" + "="*60)
    print("CONTENT MODERATION SYSTEM - FULL PIPELINE")
    print("="*60)

    # Step 1: Preprocess
    if not preprocess_data(): # Run preprocessing
        print("\nError: Preprocessing failed!")
        return

    # Step 2: Train
    if not train_model(): # Run training
        print("\nError: Training failed!")
        return

    # Step 3: Run API
    print("\n" + "="*60)
    print("✓ ALL STEPS COMPLETE!")
    print("="*60)
    print("\nNow starting API server...")
    print("Open another terminal and run:")
    print("  cd frontend")
    print("  npm install")
    print("  npm run dev")
    print()

    run_api() # Run API

if __name__ == "__main__":
    main() # Run main
