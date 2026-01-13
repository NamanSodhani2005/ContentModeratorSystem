"""
Gymnasium environment for content moderation.

State: [768-dim comment embedding, 10-dim user history, 5-dim platform metrics]
Actions: {0: keep, 1: warn, 2: remove, 3: temp_ban, 4: perma_ban}
Reward: Multi-objective (toxicity reduction, false positives, user retention, fairness)
"""

import numpy as np # Array operations
import gymnasium as gym # RL environment framework
from gymnasium import spaces # Space definitions
from enum import IntEnum # Integer enumerations

class ModerationAction(IntEnum): # Action enumeration
    KEEP = 0 # No action
    WARN = 1 # Issue warning
    REMOVE = 2 # Remove content
    TEMP_BAN = 3 # Temporary ban
    PERMA_BAN = 4 # Permanent ban

class ForumEnvironment(gym.Env):
    """
    Content moderation RL environment simulating an online forum.
    """

    metadata = {'render_modes': []} # No rendering

    def __init__(self, embeddings, labels, max_steps=500): # Initialize environment
        """
        Args:
            embeddings: np.array of shape (N, 768) - DistilBERT embeddings
            labels: np.array of shape (N, 6) - toxicity labels
            max_steps: Maximum steps per episode
        """
        super().__init__() # Call parent constructor

        self.embeddings = embeddings # Comment embeddings
        self.labels = labels # Toxicity labels
        self.max_steps = max_steps # Episode length

        # State: [comment_embedding(768), user_history(10), platform_metrics(5)]
        self.observation_space = spaces.Box( # Define state space
            low=-np.inf, # Minimum value
            high=np.inf, # Maximum value
            shape=(783,), # 783-dimensional state
            dtype=np.float32 # Float32 precision
        )

        # Actions: 5 moderation decisions
        self.action_space = spaces.Discrete(5) # 5 discrete actions

        # Initialize state
        self.current_step = 0 # Step counter
        self.platform_health = 1.0 # Platform health metric
        self.user_satisfaction = 1.0 # User satisfaction metric
        self.false_positive_count = 0 # False positive counter
        self.total_actions = 0 # Total actions taken

        # User simulator state
        self.user_history = np.zeros(10) # User history vector

        self.reset() # Reset environment

    def reset(self, seed=None, options=None): # Reset environment
        """Reset environment to initial state."""
        super().reset(seed=seed) # Call parent reset

        self.current_step = 0 # Reset step counter
        self.platform_health = 1.0 # Reset platform health
        self.user_satisfaction = 1.0 # Reset user satisfaction
        self.false_positive_count = 0 # Reset FP counter
        self.total_actions = 0 # Reset action counter

        # Reset user history
        self.user_history = np.array([ # Initialize user history
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

        # Sample initial comment
        self.current_idx = np.random.randint(0, len(self.embeddings)) # Random comment
        self.current_embedding = self.embeddings[self.current_idx] # Get embedding
        self.current_toxicity = self.labels[self.current_idx] if self.labels is not None else np.random.rand(6) # Get toxicity

        state = self._get_state() # Build state vector
        return state, {} # Return state

    def _get_state(self): # Build state vector
        """Construct state vector from current comment and context."""
        # Platform metrics: [health, satisfaction, false_positive_rate, moderation_rate, time_step_norm]
        platform_metrics = np.array([ # Calculate platform metrics
            self.platform_health, # Current health
            self.user_satisfaction, # Current satisfaction
            self.false_positive_count / max(1, self.total_actions), # FP rate
            self.total_actions / max(1, self.current_step), # Moderation rate
            self.current_step / self.max_steps # Progress in episode
        ])

        state = np.concatenate([ # Combine all components
            self.current_embedding, # Comment embedding
            self.user_history, # User history
            platform_metrics # Platform metrics
        ]).astype(np.float32) # Convert to float32

        return state # Return state vector

    def step(self, action): # Take environment step
        """Execute moderation action and return next state, reward, done, info."""
        self.current_step += 1 # Increment step counter
        self.total_actions += 1 # Increment action counter

        # Get toxicity of current comment
        is_toxic = self.current_toxicity[0] > 0.5 # Binary toxicity threshold
        toxicity_score = self.current_toxicity[0] # Toxicity score

        # Calculate reward components
        reward = self._calculate_reward(action, toxicity_score, is_toxic) # Compute reward

        # Update user history and platform state
        self._update_state(action, is_toxic) # Update metrics

        # Sample next comment
        self.current_idx = np.random.randint(0, len(self.embeddings)) # Random next comment
        self.current_embedding = self.embeddings[self.current_idx] # Get embedding
        self.current_toxicity = self.labels[self.current_idx] if self.labels is not None else np.random.rand(6) # Get toxicity

        # Check if episode is done
        done = self.current_step >= self.max_steps or self.platform_health <= 0.3 # Episode termination condition

        state = self._get_state() # Build next state
        info = { # Info dictionary
            'platform_health': self.platform_health, # Current health
            'user_satisfaction': self.user_satisfaction, # Current satisfaction
            'false_positive_rate': self.false_positive_count / max(1, self.total_actions), # FP rate
            'action_taken': ModerationAction(action).name # Action name
        }

        return state, reward, done, False, info # Return step results

    def _calculate_reward(self, action, toxicity_score, is_toxic): # Calculate reward
        """
        Multi-objective reward function.

        Components:
        1. Toxicity reduction: Reward removing actual toxic content
        2. False positive penalty: Penalize over-moderation
        3. User retention: Penalize harsh actions on benign users
        4. Platform health: Long-term community quality
        """
        reward = 0.0 # Initialize reward

        action_severity = [0, 0.2, 0.5, 0.8, 1.0] # Action severity levels

        if is_toxic: # Toxic content
            # Correct detection: reward scales with action severity and toxicity
            if action in [ModerationAction.WARN, ModerationAction.REMOVE, ModerationAction.TEMP_BAN]: # Appropriate actions
                reward += 1.0 + toxicity_score # Reward correct moderation
            elif action == ModerationAction.PERMA_BAN: # Perma ban
                # Only reward perma-ban for very toxic content
                reward += 2.0 if toxicity_score > 0.8 else -0.5 # Reward extreme toxicity only
            else: # Keep action
                # Missed toxic content
                reward -= 1.5 * toxicity_score # Penalize missed toxicity
        else: # Non-toxic content
            # Non-toxic content
            if action == ModerationAction.KEEP: # Keep action
                reward += 0.5 # Small reward
            else: # Moderation action
                # False positive: penalize severity
                self.false_positive_count += 1 # Increment FP counter
                reward -= 1.0 * action_severity[action] # Penalize false positive

        # Platform health bonus
        reward += 0.1 * self.platform_health # Health bonus

        # User retention penalty for harsh actions
        if action >= ModerationAction.TEMP_BAN: # Ban actions
            reward -= 0.2 * self.user_satisfaction # User retention penalty

        return reward # Return total reward

    def _update_state(self, action, is_toxic): # Update environment state
        """Update platform metrics and user history based on action."""
        # Update user history
        if action == ModerationAction.WARN: # Warning action
            self.user_history[1] += 1 # Increment warnings
        elif action == ModerationAction.REMOVE: # Remove action
            self.user_history[2] += 1 # Increment removals
        elif action == ModerationAction.TEMP_BAN: # Temp ban
            self.user_history[3] += 1 # Increment temp bans
        elif action == ModerationAction.PERMA_BAN: # Perma ban
            self.user_history[4] += 1 # Increment perma bans

        # Update platform health based on moderation quality
        if is_toxic and action != ModerationAction.KEEP: # Good moderation
            self.platform_health += 0.01 # Increase health
        elif not is_toxic and action in [ModerationAction.TEMP_BAN, ModerationAction.PERMA_BAN]: # False positive
            self.platform_health -= 0.02 # Decrease health

        # Update user satisfaction
        if action == ModerationAction.KEEP or (is_toxic and action in [ModerationAction.WARN, ModerationAction.REMOVE]): # Fair moderation
            self.user_satisfaction += 0.005 # Increase satisfaction
        else: # Harsh or unfair
            self.user_satisfaction -= 0.01 # Decrease satisfaction

        # Clamp values
        self.platform_health = np.clip(self.platform_health, 0.0, 1.0) # Clamp health
        self.user_satisfaction = np.clip(self.user_satisfaction, 0.0, 1.0) # Clamp satisfaction

    def render(self): # Render environment
        """Optional rendering (not implemented)."""
        pass # No rendering

    def close(self): # Close environment
        """Cleanup."""
        pass # No cleanup needed
