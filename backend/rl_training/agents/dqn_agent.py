"""
DQN agent with Double DQN, soft target updates, and epsilon-greedy exploration.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for off-policy learning.

    Stores (s, a, r, s', done) transitions and provides uniform random
    sampling. Using replay breaks temporal correlations between consecutive
    samples, which stabilizes DQN training.
    """

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class NumpyReplayBuffer:
    """Pre-allocated numpy replay buffer — much faster than deque of tuples.

    Advantages over the deque-based ReplayBuffer:
      - push_batch(): insert N transitions in one numpy copy (for vectorized envs)
      - sample(): numpy fancy indexing instead of random.sample + zip + np.array
      - No Python object overhead per transition
      - ~5-10x faster for large buffers
    """

    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        """Store a single transition."""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, states, actions, rewards, next_states, dones):
        """Store N transitions at once (single numpy copy).

        Handles wrap-around when the batch straddles the end of the buffer.
        """
        n = len(states)
        if self.pos + n <= self.capacity:
            # Fast path: no wrap-around
            s = slice(self.pos, self.pos + n)
            self.states[s] = states
            self.actions[s] = actions
            self.rewards[s] = rewards
            self.next_states[s] = next_states
            self.dones[s] = dones
            self.pos = (self.pos + n) % self.capacity
        else:
            # Handle wrap-around: fill to end, then start from 0
            first = self.capacity - self.pos
            self.states[self.pos:] = states[:first]
            self.actions[self.pos:] = actions[:first]
            self.rewards[self.pos:] = rewards[:first]
            self.next_states[self.pos:] = next_states[:first]
            self.dones[self.pos:] = dones[:first]

            rest = n - first
            self.states[:rest] = states[first:]
            self.actions[:rest] = actions[first:]
            self.rewards[:rest] = rewards[first:]
            self.next_states[:rest] = next_states[first:]
            self.dones[:rest] = dones[first:]
            self.pos = rest
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        """Sample a random batch. Uses numpy fancy indexing — no Python loop."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size


class DQNAgent:
    """Double DQN agent for content moderation.

    Key components:
      - Policy network: predicts Q(s, a) for all actions
      - Target network: provides stable Q-value targets (updated via soft copy)
      - Replay buffer: stores transitions for experience replay
      - Epsilon-greedy: explores randomly with probability epsilon, decayed over time
    """

    def __init__(self, policy_network, target_network, device='cpu',
                 lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay=0.995, buffer_capacity=100000, state_dim=None):

        self.policy_network = policy_network.to(device)
        self.target_network = target_network.to(device)
        self.device = device

        # Initialize target network with same weights as policy network
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()  # Target network is never trained directly

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Use pre-allocated numpy buffer if state_dim is known (training),
        # otherwise fall back to deque-based buffer (inference / backward compat)
        if state_dim is not None:
            self.replay_buffer = NumpyReplayBuffer(buffer_capacity, state_dim)
        else:
            self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Huber loss (SmoothL1) is less sensitive to outliers than MSE
        self.criterion = nn.SmoothL1Loss()

    def select_action(self, state, eval_mode=False):
        """Pick an action using epsilon-greedy policy (single state).

        Returns (action, q_values) where q_values is a numpy array of
        Q-values for all 5 moderation actions.
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_network(state_tensor)
            q_np = q_values.cpu().numpy()[0]

        # Epsilon-greedy: random action with prob epsilon, greedy otherwise
        if not eval_mode and random.random() < self.epsilon:
            action = np.random.randint(0, 5)
        else:
            action = int(np.argmax(q_np))

        return action, q_np

    def select_actions_batch(self, states):
        """Pick actions for N states at once (single GPU forward pass).

        Args:
            states: (N, state_dim) float32 numpy array
        Returns:
            actions: (N,) int numpy array
            q_values: (N, num_actions) float32 numpy array
        """
        with torch.no_grad():
            # from_numpy avoids a copy when input is already float32 C-contiguous
            state_tensor = torch.from_numpy(states).to(self.device)
            q_values = self.policy_network(state_tensor)
            q_np = q_values.cpu().numpy()

        # Vectorized epsilon-greedy: random mask applied across all envs
        actions = np.argmax(q_np, axis=1)
        random_mask = np.random.rand(len(states)) < self.epsilon
        n_random = random_mask.sum()
        if n_random > 0:
            actions[random_mask] = np.random.randint(0, q_np.shape[1], size=n_random)

        return actions, q_np

    def train_step(self, batch_size=128):
        """One gradient step on a random batch from the replay buffer.

        Uses Double DQN: the policy network selects the best next action,
        but the target network evaluates its Q-value. This decouples action
        selection from evaluation, reducing overestimation bias.
        """
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for the actions that were actually taken
        q_values = self.policy_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target computation:
        # 1. Policy network picks the best next action
        # 2. Target network evaluates Q-value of that action
        with torch.no_grad():
            next_q_policy = self.policy_network(next_states)
            best_actions = next_q_policy.argmax(1, keepdim=True)
            next_q_target = self.target_network(next_states)
            next_q = next_q_target.gather(1, best_actions).squeeze(1)
            # Bellman target: r + gamma * Q_target(s', argmax_a Q_policy(s', a))
            targets = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents explosive updates from rare large rewards
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self, tau=0.005):
        """Soft update: target = tau * policy + (1 - tau) * target.

        Polyak averaging with small tau (0.005) gives smoother target updates
        than periodic hard copies, improving training stability.
        """
        for target_param, policy_param in zip(self.target_network.parameters(),
                                              self.policy_network.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

    def decay_epsilon(self):
        """Multiplicative epsilon decay, clamped to epsilon_end."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {path}")
