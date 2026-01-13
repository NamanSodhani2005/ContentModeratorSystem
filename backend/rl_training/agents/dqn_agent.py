"""
DQN Agent with experience replay and target network.
"""

import torch # PyTorch framework
import torch.nn as nn # Neural network modules
import torch.optim as optim # Optimization algorithms
import numpy as np # Array operations
from collections import deque # Double-ended queue
import random # Random sampling

class ReplayBuffer: # Experience replay buffer
    """Experience replay buffer for off-policy learning."""

    def __init__(self, capacity=100000): # Initialize buffer
        self.buffer = deque(maxlen=capacity) # Create circular buffer

    def push(self, state, action, reward, next_state, done): # Store transition
        """Store transition."""
        self.buffer.append((state, action, reward, next_state, done)) # Add to buffer

    def sample(self, batch_size): # Sample batch
        """Sample random batch of transitions."""
        batch = random.sample(self.buffer, batch_size) # Random sample
        states, actions, rewards, next_states, dones = zip(*batch) # Unzip batch

        return ( # Return as arrays
            np.array(states), # States array
            np.array(actions), # Actions array
            np.array(rewards), # Rewards array
            np.array(next_states), # Next states array
            np.array(dones) # Done flags
        )

    def __len__(self): # Get buffer size
        return len(self.buffer) # Return size

class DQNAgent: # DQN agent class
    """
    Deep Q-Network agent for content moderation.

    Features:
    - Experience replay
    - Target network with soft updates
    - Epsilon-greedy exploration
    - Gradient clipping
    """

    def __init__(self, policy_network, target_network, device='cpu', # Initialize agent
                 lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay=0.995, buffer_capacity=100000):

        self.policy_network = policy_network.to(device) # Move policy to device
        self.target_network = target_network.to(device) # Move target to device
        self.device = device # Store device

        # Copy weights to target network
        self.target_network.load_state_dict(self.policy_network.state_dict()) # Copy weights
        self.target_network.eval() # Set eval mode

        # Training parameters
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr) # Adam optimizer
        self.gamma = gamma # Discount factor
        self.epsilon = epsilon_start # Current exploration rate
        self.epsilon_end = epsilon_end # Minimum exploration rate
        self.epsilon_decay = epsilon_decay # Decay factor

        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_capacity) # Create replay buffer

        # Loss function
        self.criterion = nn.MSELoss() # MSE loss

    def select_action(self, state, eval_mode=False): # Select action
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            eval_mode: If True, use greedy policy (no exploration)
        Returns:
            action: Selected action
            q_values: Q-values for all actions
            attention_weights: Attention weights for interpretability
        """
        if not eval_mode and random.random() < self.epsilon: # Exploration check
            # Explore: random action
            action = np.random.randint(0, 5) # Random action
            with torch.no_grad(): # No gradients
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) # Convert to tensor
                q_values, attention_weights = self.policy_network(state_tensor) # Forward pass
            return action, q_values.cpu().numpy()[0], attention_weights.cpu().numpy()[0] # Return random action
        else: # Exploitation
            # Exploit: best action
            with torch.no_grad(): # No gradients
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device) # Convert to tensor
                q_values, attention_weights = self.policy_network(state_tensor) # Forward pass
                action = torch.argmax(q_values, dim=-1).item() # Select best action
            return action, q_values.cpu().numpy()[0], attention_weights.cpu().numpy()[0] # Return best action

    def train_step(self, batch_size=128): # Training step
        """
        Perform one training step using experience replay.

        Returns:
            loss: Training loss
        """
        if len(self.replay_buffer) < batch_size: # Check buffer size
            return None # Skip training

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size) # Sample experiences

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device) # States to tensor
        actions = torch.LongTensor(actions).to(self.device) # Actions to tensor
        rewards = torch.FloatTensor(rewards).to(self.device) # Rewards to tensor
        next_states = torch.FloatTensor(next_states).to(self.device) # Next states to tensor
        dones = torch.FloatTensor(dones).to(self.device) # Dones to tensor

        # Compute Q(s, a)
        q_values, _ = self.policy_network(states) # Forward pass policy network
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1) # Gather selected action Q-values

        # Compute target: r + γ * max_a' Q_target(s', a')
        with torch.no_grad(): # Disable gradient computation
            next_q_values, _ = self.target_network(next_states) # Forward pass target network
            max_next_q = next_q_values.max(1)[0] # Get max Q-value
            targets = rewards + (1 - dones) * self.gamma * max_next_q # Bellman equation target

        # Compute loss
        loss = self.criterion(q_values, targets) # Compute MSE loss

        # Optimize
        self.optimizer.zero_grad() # Zero gradients
        loss.backward() # Compute gradients
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0) # Clip gradients
        self.optimizer.step() # Update network weights

        return loss.item() # Return scalar loss

    def update_target_network(self, tau=0.005): # Update target network
        """Soft update of target network: θ' ← τ*θ + (1-τ)*θ'"""
        for target_param, policy_param in zip(self.target_network.parameters(), # Iterate over parameters
                                              self.policy_network.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data) # Soft update weights

    def decay_epsilon(self): # Decay exploration
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay) # Apply decay

    def save(self, path): # Save model
        """Save agent state."""
        torch.save({ # Create checkpoint dictionary
            'policy_state_dict': self.policy_network.state_dict(), # Policy network weights
            'target_state_dict': self.target_network.state_dict(), # Target network weights
            'optimizer_state_dict': self.optimizer.state_dict(), # Optimizer state
            'epsilon': self.epsilon # Current epsilon value
        }, path) # Write to file
        print(f"✓ Model saved to {path}") # Print confirmation

    def load(self, path): # Load model
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device) # Load checkpoint file
        self.policy_network.load_state_dict(checkpoint['policy_state_dict']) # Restore policy weights
        self.target_network.load_state_dict(checkpoint['target_state_dict']) # Restore target weights
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # Restore optimizer state
        self.epsilon = checkpoint['epsilon'] # Restore epsilon value
        print(f"✓ Model loaded from {path}") # Print confirmation
