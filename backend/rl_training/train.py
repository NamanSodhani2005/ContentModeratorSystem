"""
Training loop for content moderation DQN agent.
"""

import os # File operations
import sys # System operations
import numpy as np # Array operations
import torch # PyTorch framework
from pathlib import Path # Path utilities
from tqdm import tqdm # Progress bars

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent)) # Add parent to path

from rl_training.environment.forum_env import ForumEnvironment # RL environment
from rl_training.models.policy_network import PolicyNetwork # Q-network
from rl_training.agents.dqn_agent import DQNAgent # DQN agent

def train( # Training function
    embeddings_path='backend/data/embeddings.npy', # Embeddings path
    labels_path='backend/data/labels.npy', # Labels path
    hate_scores_path='backend/data/hate_scores.npy', # Hate scores path
    num_episodes=1000, # Training episodes
    max_steps=500, # Steps per episode
    batch_size=128, # Batch size
    save_interval=100, # Save interval
    device='cpu' # Device
):
    """
    Train DQN agent on content moderation task.

    Args:
        embeddings_path: Path to comment embeddings
        labels_path: Path to toxicity labels
        hate_scores_path: Path to hate speech scores
        num_episodes: Number of training episodes
        max_steps: Steps per episode
        batch_size: Training batch size
        save_interval: Save checkpoint every N episodes
        device: 'cpu' or 'cuda'
    """
    print("=" * 60)
    print("Content Moderation RL Training")
    print("=" * 60)

    # Load data
    print(f"\nLoading data...")
    embeddings = np.load(embeddings_path) # Load embeddings
    labels = np.load(labels_path) # Load labels
    hate_scores = np.load(hate_scores_path) if os.path.exists(hate_scores_path) else None
    print(f"✓ Loaded {len(embeddings)} comments")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Labels shape: {labels.shape}")

    # Initialize environment
    print(f"\nInitializing environment...")
    env = ForumEnvironment(embeddings, labels, hate_scores=hate_scores, max_steps=max_steps) # Create environment
    print(f"✓ Environment created")
    print(f"  State space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n}")

    # Initialize networks
    print(f"\nInitializing neural networks...")
    policy_network = PolicyNetwork() # Policy network
    target_network = PolicyNetwork() # Target network
    print(f"✓ Networks created")
    print(f"  Parameters: {sum(p.numel() for p in policy_network.parameters()):,}")

    # Initialize agent
    print(f"\nInitializing DQN agent...")
    agent = DQNAgent( # Create agent
        policy_network=policy_network, # Policy network
        target_network=target_network, # Target network
        device=device, # Device
        lr=1e-4, # Learning rate
        gamma=0.99, # Discount factor
        epsilon_start=1.0, # Initial epsilon
        epsilon_end=0.05, # Final epsilon
        epsilon_decay=0.995 # Epsilon decay
    )
    print(f"✓ Agent ready")

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {num_episodes} episodes")
    print(f"{'='*60}\n")

    training_stats = { # Statistics dictionary
        'episode_rewards': [], # Reward per episode
        'episode_lengths': [], # Steps per episode
        'losses': [], # Loss per episode
        'epsilon_values': [], # Epsilon per episode
        'platform_health': [], # Health per episode
        'false_positive_rates': [] # FP rate per episode
    }

    for episode in range(num_episodes): # Episode loop
        state, _ = env.reset() # Reset environment
        episode_reward = 0 # Cumulative reward
        episode_loss = [] # Episode losses

        for step in range(max_steps): # Step loop
            # Select action
            action, q_values, attention = agent.select_action(state) # Select action

            # Execute action
            next_state, reward, done, truncated, info = env.step(action) # Take step

            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done) # Store experience

            # Train
            loss = agent.train_step(batch_size) # Training step
            if loss is not None: # If training occurred
                episode_loss.append(loss) # Record loss

            episode_reward += reward # Accumulate reward
            state = next_state # Update state

            if done: # Episode done
                break # Exit step loop

        # Update target network
        if episode % 10 == 0: # Every 10 episodes
            agent.update_target_network(tau=0.005) # Soft update

        # Decay epsilon
        agent.decay_epsilon() # Decay exploration

        # Record statistics
        training_stats['episode_rewards'].append(episode_reward) # Record reward
        training_stats['episode_lengths'].append(step + 1) # Record length
        training_stats['losses'].append(np.mean(episode_loss) if episode_loss else 0) # Record loss
        training_stats['epsilon_values'].append(agent.epsilon) # Record epsilon
        training_stats['platform_health'].append(info['platform_health']) # Record health
        training_stats['false_positive_rates'].append(info['false_positive_rate']) # Record FP rate

        # Print progress
        if episode % 10 == 0: # Every 10 episodes
            avg_reward = np.mean(training_stats['episode_rewards'][-10:]) # Average reward
            avg_loss = np.mean(training_stats['losses'][-10:]) # Average loss
            print(f"Episode {episode:4d} | " # Print stats
                  f"Reward: {avg_reward:7.2f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Health: {info['platform_health']:.2f} | "
                  f"FP: {info['false_positive_rate']:.3f}")

        # Save checkpoint
        if episode % save_interval == 0 and episode > 0: # Every N episodes
            save_dir = Path('backend/saved_models') # Save directory
            save_dir.mkdir(exist_ok=True) # Create if needed
            save_path = save_dir / f'dqn_checkpoint_ep{episode}.pt' # Checkpoint path
            agent.save(save_path) # Save checkpoint

    # Save final model
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    save_dir = Path('backend/saved_models') # Save directory
    save_dir.mkdir(exist_ok=True) # Create if needed
    final_path = save_dir / 'dqn_final.pt' # Final model path
    agent.save(final_path) # Save final model

    # Save training stats
    stats_path = save_dir / 'training_stats.npz' # Stats path
    np.savez(stats_path, **training_stats) # Save statistics
    print(f"✓ Training statistics saved to {stats_path}")

    # Print final statistics
    print(f"\nFinal Statistics:")
    print(f"  Average reward (last 100): {np.mean(training_stats['episode_rewards'][-100:]):.2f}") # Avg reward
    print(f"  Average loss (last 100): {np.mean(training_stats['losses'][-100:]):.4f}") # Avg loss
    print(f"  Final epsilon: {agent.epsilon:.3f}") # Final epsilon
    print(f"  Final platform health: {training_stats['platform_health'][-1]:.2f}") # Final health
    print(f"  Final false positive rate: {training_stats['false_positive_rates'][-1]:.3f}") # Final FP rate

def main(): # Main function
    # Check for CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # Select device
    print(f"Using device: {device}")

    # Check if data exists
    embeddings_path = 'backend/data/embeddings.npy' # Embeddings path
    labels_path = 'backend/data/labels.npy' # Labels path

    if not os.path.exists(embeddings_path): # Check file exists
        print("Error: Embeddings not found. Please run:")
        print("  1. python backend/data/download.py")
        print("  2. python backend/data/preprocess.py")
        sys.exit(1) # Exit with error

    # Start training
    train( # Run training
        embeddings_path=embeddings_path, # Embeddings path
        labels_path=labels_path, # Labels path
        num_episodes=1000, # 1000 episodes
        max_steps=500, # 500 steps each
        batch_size=128, # Batch of 128
        save_interval=100, # Save every 100
        device=device # Device
    )

if __name__ == "__main__":
    main() # Run main function
