"""
DQN agent.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Replay buffer."""

    # Initialize a fixed-size deque to store transitions for replay.
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    # Append a transition tuple for later sampling.
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # Randomly sample a batch of stored transitions for training.
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

    # Return current number of transitions in the buffer.
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN moderation agent."""

    # Initialize networks, optimizer, replay buffer, and exploration parameters.
    def __init__(self, policy_network, target_network, device='cpu',
                 lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay=0.995, buffer_capacity=100000):

        self.policy_network = policy_network.to(device)
        self.target_network = target_network.to(device)
        self.device = device

        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()  # eval mode

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)  # optimizer
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.replay_buffer = ReplayBuffer(buffer_capacity)  # replay buffer

        self.criterion = nn.SmoothL1Loss()  # Huber loss

    # Choose an action using epsilon-greedy policy and return values and attention.
    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            action = np.random.randint(0, 5)  # explore action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values, attention_weights = self.policy_network(state_tensor)
            return action, q_values.cpu().numpy()[0], attention_weights.cpu().numpy()[0]

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values, attention_weights = self.policy_network(state_tensor)
            action = torch.argmax(q_values, dim=-1).item()  # exploit action
        return action, q_values.cpu().numpy()[0], attention_weights.cpu().numpy()[0]

    # Train on a replay batch and update network weights if enough data.
    def train_step(self, batch_size=128):
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values, _ = self.policy_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_policy, _ = self.policy_network(next_states)
            best_actions = next_q_policy.argmax(1, keepdim=True)
            next_q_target, _ = self.target_network(next_states)
            next_q = next_q_target.gather(1, best_actions).squeeze(1)
            targets = rewards + (1 - dones) * self.gamma * next_q  # bellman target

        loss = self.criterion(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)  # clip grads
        self.optimizer.step()

        return loss.item()

    # Soft-update target network parameters toward policy network values.
    def update_target_network(self, tau=0.005):
        for target_param, policy_param in zip(self.target_network.parameters(),
                                              self.policy_network.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

    # Decay exploration rate toward the minimum bound.
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # Save agent and optimizer state to disk for checkpointing.
    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"Model saved to {path}")

    # Load agent and optimizer state from disk for resuming training.
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {path}")
