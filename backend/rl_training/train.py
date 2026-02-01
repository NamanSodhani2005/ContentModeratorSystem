"""
DQN training loop.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from rl_training.environment.forum_env import ForumEnvironment
from rl_training.models.policy_network import PolicyNetwork
from rl_training.agents.dqn_agent import DQNAgent


# Run the DQN training loop over episodes, saving checkpoints and metrics.
def train(
    embeddings_path='backend/data/embeddings.npy',
    labels_path='backend/data/labels.npy',
    hate_scores_path='backend/data/hate_scores.npy',
    target_features_path='backend/data/target_features.npy',
    target_toxicity_path='backend/data/target_toxicity.npy',
    num_episodes=1000,
    max_steps=500,
    batch_size=128,
    save_interval=100,
    device='cpu'
):
    print("=" * 60)
    print("Content Moderation RL Training")
    print("=" * 60)

    print("\nLoading data...")
    embeddings = np.load(embeddings_path)  # load embeddings
    labels = np.load(labels_path)  # load labels
    hate_scores = np.load(hate_scores_path) if os.path.exists(hate_scores_path) else None
    target_features = np.load(target_features_path) if os.path.exists(target_features_path) else None
    target_toxicity = np.load(target_toxicity_path) if os.path.exists(target_toxicity_path) else None
    print(f"Loaded {len(embeddings)} comments")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Labels shape: {labels.shape}")
    if target_features is not None:
        print(f"  Target features shape: {target_features.shape}")
    if target_toxicity is not None:
        print(f"  Target toxicity shape: {target_toxicity.shape}")

    print("\nInitializing environment...")
    env = ForumEnvironment(
        embeddings,
        labels,
        hate_scores=hate_scores,
        target_features=target_features,
        target_toxicity=target_toxicity,
        max_steps=max_steps
    )
    print("Environment created")
    print(f"  State space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n}")

    print("\nInitializing neural networks...")
    policy_network = PolicyNetwork(context_dim=22)
    target_network = PolicyNetwork(context_dim=22)
    print("Networks created")
    print(f"  Parameters: {sum(p.numel() for p in policy_network.parameters()):,}")

    print("\nInitializing DQN agent...")
    agent = DQNAgent(
        policy_network=policy_network,
        target_network=target_network,
        device=device,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995
    )
    print("Agent ready")

    print(f"\n{'='*60}")
    print(f"Starting training for {num_episodes} episodes")
    print(f"{'='*60}\n")

    training_stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'losses': [],
        'epsilon_values': [],
        'platform_health': [],
        'false_positive_rates': []
    }

    interrupted = False
    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_loss = []

            for step in range(max_steps):
                action, q_values, attention = agent.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                agent.replay_buffer.push(state, action, reward, next_state, done)

                loss = agent.train_step(batch_size)
                if loss is not None:
                    episode_loss.append(loss)

                episode_reward += reward
                state = next_state

                if done:
                    break

            if episode % 10 == 0:
                agent.update_target_network(tau=0.005)  # sync target

            agent.decay_epsilon()

            training_stats['episode_rewards'].append(episode_reward)
            training_stats['episode_lengths'].append(step + 1)
            training_stats['losses'].append(np.mean(episode_loss) if episode_loss else 0)
            training_stats['epsilon_values'].append(agent.epsilon)
            training_stats['platform_health'].append(info['platform_health'])
            training_stats['false_positive_rates'].append(info['false_positive_rate'])

            if episode % 10 == 0:
                avg_reward = np.mean(training_stats['episode_rewards'][-10:])
                avg_loss = np.mean(training_stats['losses'][-10:])
                print(f"Episode {episode:4d} | "
                      f"Reward: {avg_reward:7.2f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Eps: {agent.epsilon:.3f} | "
                      f"Health: {info['platform_health']:.2f} | "
                      f"FP: {info['false_positive_rate']:.3f}")

            if episode % save_interval == 0 and episode > 0:
                save_dir = Path('backend/saved_models')
                save_dir.mkdir(exist_ok=True)
                save_path = save_dir / f'dqn_checkpoint_ep{episode}.pt'
                agent.save(save_path)  # checkpoint
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted by user. Saving current model and stats...")

    print("\n" + "=" * 60)
    if interrupted:
        print("Training interrupted.")
    else:
        print("Training complete!")
    print("=" * 60)

    save_dir = Path('backend/saved_models')
    save_dir.mkdir(exist_ok=True)
    final_path = save_dir / 'dqn_final.pt'
    agent.save(final_path)  # save model

    stats_path = save_dir / 'training_stats.npz'
    np.savez(stats_path, **training_stats)  # save stats
    print(f"Training statistics saved to {stats_path}")

    if training_stats['episode_rewards']:
        print(f"\nFinal Statistics:")
        print(f"  Average reward (last 100): {np.mean(training_stats['episode_rewards'][-100:]):.2f}")
        print(f"  Average loss (last 100): {np.mean(training_stats['losses'][-100:]):.4f}")
        print(f"  Final epsilon: {agent.epsilon:.3f}")
        print(f"  Final platform health: {training_stats['platform_health'][-1]:.2f}")
        print(f"  Final false positive rate: {training_stats['false_positive_rates'][-1]:.3f}")
    else:
        print("\nNo completed episodes to summarize.")


# Select device, validate data files, and start training.
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # select device
    print(f"Using device: {device}")

    embeddings_path = 'backend/data/embeddings.npy'
    labels_path = 'backend/data/labels.npy'

    if not os.path.exists(embeddings_path):
        print("Error: Embeddings not found. Please run:")
        print("  1. python backend/data/download.py")
        print("  2. python backend/data/preprocess.py")
        sys.exit(1)

    train(
        embeddings_path=embeddings_path,
        labels_path=labels_path,
        num_episodes=1000,
        max_steps=500,
        batch_size=128,
        save_interval=100,
        device=device
    )


if __name__ == "__main__":
    main()
