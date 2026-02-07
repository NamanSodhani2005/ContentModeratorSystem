"""
DQN training loop for the content moderation agent.

Uses vectorized environments (N envs stepped in parallel via numpy) and
batched GPU forward passes for significant speedup over the naive
single-env loop.
"""

import os
import sys
import time
from collections import Counter

import numpy as np
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from rl_training.environment.forum_env import VectorizedForumEnv
from rl_training.models.policy_network import PolicyNetwork
from rl_training.agents.dqn_agent import DQNAgent


def train(
    embeddings_path='backend/data/embeddings.npy',
    labels_path='backend/data/labels.npy',
    target_features_path='backend/data/target_features.npy',
    target_toxicity_path='backend/data/target_toxicity.npy',
    num_episodes=1000,
    max_steps=500,
    batch_size=128,
    save_interval=100,
    train_every=4,
    num_envs=16,
    device='cpu'
):
    print("=" * 60)
    print("Content Moderation RL Training")
    print("=" * 60)

    # --- Hyperparameter summary ---
    print("\nHyperparameters:")
    print(f"  Episodes:        {num_episodes}")
    print(f"  Max steps/ep:    {max_steps}")
    print(f"  Batch size:      {batch_size}")
    print(f"  Train every:     {train_every} steps")
    print(f"  Parallel envs:   {num_envs}")
    print(f"  Save interval:   {save_interval} episodes")
    print(f"  Device:          {device}")

    # --- Phase 1: Load preprocessed data ---
    print("\nLoading data...")
    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)
    target_features = np.load(target_features_path) if os.path.exists(target_features_path) else None
    target_toxicity = np.load(target_toxicity_path) if os.path.exists(target_toxicity_path) else None
    print(f"Loaded {len(embeddings)} comments")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Labels shape: {labels.shape}")
    if target_features is not None:
        print(f"  Target features shape: {target_features.shape}")
    if target_toxicity is not None:
        print(f"  Target toxicity shape: {target_toxicity.shape}")

    # --- Phase 2: Create vectorized environment (N envs, shared data, no copies) ---
    print(f"\nInitializing {num_envs} parallel environments...")
    vec_env = VectorizedForumEnv(
        embeddings,
        labels,
        target_features=target_features,
        target_toxicity=target_toxicity,
        max_steps=max_steps,
        num_envs=num_envs
    )
    print(f"  State dim: {vec_env.state_dim}")
    print(f"  Actions: {vec_env.num_actions}")

    # --- Phase 3: Build policy + target networks ---
    print("\nInitializing neural networks...")
    policy_network = PolicyNetwork()
    target_network = PolicyNetwork()
    param_count = sum(p.numel() for p in policy_network.parameters())
    print(f"  Parameters: {param_count:,}")

    # TF32 tensor cores: faster float32 matmul on Ampere+ GPUs (RTX 30xx/40xx/50xx)
    if device == 'cuda':
        torch.set_float32_matmul_precision('high')
        print("  TF32 matmul: enabled")

    # torch.compile: fuses ops and reduces Python overhead (PyTorch 2.x+)
    # Requires Triton, which is not available on Windows — skip there.
    compiled = False
    if hasattr(torch, 'compile') and sys.platform != 'win32':
        try:
            policy_network = torch.compile(policy_network)
            target_network = torch.compile(target_network)
            compiled = True
            print("  torch.compile: enabled")
        except Exception as e:
            print(f"  torch.compile: failed ({e}), continuing without")
    if not compiled:
        print(f"  torch.compile: skipped ({'Windows — no Triton' if sys.platform == 'win32' else 'not available'})")

    # --- Phase 4: Initialize DQN agent ---
    print("\nInitializing DQN agent...")
    state_dim = vec_env.state_dim
    agent = DQNAgent(
        policy_network=policy_network,
        target_network=target_network,
        device=device,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        state_dim=state_dim
    )
    print("Agent ready")
    print(f"  LR: {1e-4}, Gamma: {0.99}")
    print(f"  Epsilon: {agent.epsilon:.2f} -> {agent.epsilon_end:.2f} (decay={agent.epsilon_decay})")
    print(f"  Replay buffer: NumpyReplayBuffer ({agent.replay_buffer.capacity:,} capacity)")

    print(f"\n{'='*60}")
    print(f"Starting training for {num_episodes} episodes ({num_envs} parallel envs)")
    print(f"{'='*60}\n")

    training_stats = {
        'episode_rewards': [],
        'episode_lengths': [],
        'losses': [],
        'epsilon_values': [],
        'false_positive_rates': []
    }

    # --- Phase 5: Vectorized training loop ---
    states = vec_env.reset_all()
    interrupted = False
    completed_episodes = 0
    global_step = 0
    recent_losses = []
    action_counts = Counter()
    recent_q_values = []
    log_interval = 10
    last_log_ep = 0
    last_save_ep = 0
    train_start = time.time()

    try:
        while completed_episodes < num_episodes:
            # Batch forward pass: all N states through GPU in one call
            actions, q_values = agent.select_actions_batch(states)
            action_counts.update(actions.tolist())
            recent_q_values.append(q_values.max(axis=1).mean())

            # Vectorized env step: pure numpy, all N envs at once
            next_states, rewards, dones, infos = vec_env.step(actions)

            # Batch push: N transitions in a single numpy copy
            agent.replay_buffer.push_batch(
                states, actions, rewards, next_states, dones.astype(np.float32)
            )

            # Gradient step every train_every loop iterations
            global_step += 1
            if global_step % train_every == 0:
                loss = agent.train_step(batch_size)
                if loss is not None:
                    recent_losses.append(loss)

            # Handle completed episodes
            if dones.any():
                done_indices = np.where(dones)[0]
                for i in done_indices:
                    ep_reward = infos['episode_rewards'][i]
                    fp_rate = infos['false_positive_rates'][i]
                    ep_len = infos['episode_lengths'][i]

                    training_stats['episode_rewards'].append(float(ep_reward))
                    training_stats['episode_lengths'].append(int(ep_len))
                    training_stats['losses'].append(
                        np.mean(recent_losses[-50:]) if recent_losses else 0
                    )
                    training_stats['epsilon_values'].append(agent.epsilon)
                    training_stats['false_positive_rates'].append(float(fp_rate))

                    completed_episodes += 1
                    agent.decay_epsilon()

                # Soft-update target network when any episode completes
                agent.update_target_network(tau=0.005)

            # --- Logging ---
            if completed_episodes >= last_log_ep + log_interval and completed_episodes > 0:
                last_log_ep = (completed_episodes // log_interval) * log_interval
                avg_reward = np.mean(training_stats['episode_rewards'][-log_interval:])
                avg_loss = np.mean(recent_losses[-100:]) if recent_losses else 0
                avg_q = np.mean(recent_q_values[-100:]) if recent_q_values else 0
                avg_fp = np.mean(training_stats['false_positive_rates'][-log_interval:])
                elapsed = time.time() - train_start
                eps_per_sec = completed_episodes / max(0.1, elapsed)
                remaining = (num_episodes - completed_episodes) / max(0.01, eps_per_sec)

                dist = [action_counts.get(i, 0) for i in range(5)]
                dist_str = "/".join(str(d) for d in dist)

                print(f"Ep {completed_episodes:4d} | "
                      f"R: {avg_reward:7.2f} | "
                      f"L: {avg_loss:.4f} | "
                      f"Q: {avg_q:.2f} | "
                      f"Eps: {agent.epsilon:.3f} | "
                      f"FP: {avg_fp:.3f} | "
                      f"Buf: {len(agent.replay_buffer):6d} | "
                      f"Act: {dist_str} | "
                      f"{eps_per_sec:.1f} ep/s | "
                      f"ETA: {remaining:.0f}s")

            # --- Checkpoint saving (once per milestone) ---
            save_milestone = (completed_episodes // save_interval) * save_interval
            if save_milestone > last_save_ep and save_milestone > 0:
                last_save_ep = save_milestone
                save_dir = Path('backend/saved_models')
                save_dir.mkdir(exist_ok=True)
                save_path = save_dir / f'dqn_checkpoint_ep{save_milestone}.pt'
                agent.save(save_path)

            states = next_states

    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted by user. Saving current model and stats...")

    # --- Phase 6: Save final model and stats ---
    total_time = time.time() - train_start
    print("\n" + "=" * 60)
    if interrupted:
        print(f"Training interrupted at episode {completed_episodes}.")
    else:
        print("Training complete!")
    print(f"Total time: {total_time:.1f}s ({completed_episodes} episodes, "
          f"{completed_episodes/max(0.1, total_time):.1f} ep/s)")
    print("=" * 60)

    save_dir = Path('backend/saved_models')
    save_dir.mkdir(exist_ok=True)
    final_path = save_dir / 'dqn_final.pt'
    agent.save(final_path)

    stats_path = save_dir / 'training_stats.npz'
    np.savez(stats_path, **training_stats)
    print(f"Training statistics saved to {stats_path}")

    if training_stats['episode_rewards']:
        print(f"\nFinal Statistics:")
        print(f"  Average reward (last 100): {np.mean(training_stats['episode_rewards'][-100:]):.2f}")
        print(f"  Average loss (last 100): {np.mean(training_stats['losses'][-100:]):.4f}")
        print(f"  Final epsilon: {agent.epsilon:.3f}")
        print(f"  Final false positive rate: {training_stats['false_positive_rates'][-1]:.3f}")
    else:
        print("\nNo completed episodes to summarize.")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
