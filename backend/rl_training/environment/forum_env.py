"""
Moderation gym environment.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import IntEnum


class ModerationAction(IntEnum):
    KEEP = 0
    WARN = 1
    REMOVE = 2
    TEMP_BAN = 3
    PERMA_BAN = 4


class ForumEnvironment(gym.Env):
    """Moderation environment."""

    metadata = {'render_modes': []}

    # Set up spaces, counters, and initial state for the moderation environment.
    def __init__(self, embeddings, labels, hate_scores=None, target_features=None, target_toxicity=None, max_steps=500):
        super().__init__()

        self.embeddings = embeddings
        self.labels = labels
        self.hate_scores = hate_scores
        self.target_features = target_features
        self.target_toxicity = target_toxicity
        self.max_steps = max_steps
        self.hate_score_dim = 3
        self.target_feature_dim = 4

        self.state_dim = 768 + self.hate_score_dim + self.target_feature_dim + 10 + 5  # state size
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(5)

        self.current_step = 0
        self.platform_health = 1.0
        self.user_satisfaction = 1.0
        self.false_positive_count = 0
        self.total_actions = 0

        self.user_history = np.zeros(10)

        if self.target_toxicity is not None and len(self.target_toxicity) != len(self.embeddings):
            raise ValueError("target_toxicity must have the same length as embeddings")

        self.reset()

    # Reset counters, sample a comment, and return the initial state.
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.platform_health = 1.0
        self.user_satisfaction = 1.0
        self.false_positive_count = 0
        self.total_actions = 0

        self.user_history = np.array([
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

        self.current_idx = np.random.randint(0, len(self.embeddings))  # sample comment
        self.current_embedding = self.embeddings[self.current_idx]
        self.current_toxicity = self.labels[self.current_idx] if self.labels is not None else np.random.rand(6)
        if self.hate_scores is not None:
            self.current_hate_scores = self.hate_scores[self.current_idx]
        else:
            self.current_hate_scores = np.zeros(self.hate_score_dim, dtype=np.float32)
        if self.target_features is not None:
            self.current_target_features = self.target_features[self.current_idx]
        else:
            self.current_target_features = np.zeros(self.target_feature_dim, dtype=np.float32)
        if self.target_toxicity is not None:
            self.current_target_toxicity = float(self.target_toxicity[self.current_idx])
        else:
            self.current_target_toxicity = None

        state = self._get_state()
        return state, {}

    # Build the state vector from embeddings, context, and platform metrics.
    def _get_state(self):
        platform_metrics = np.array([
            self.platform_health,
            self.user_satisfaction,
            self.false_positive_count / max(1, self.total_actions),
            self.total_actions / max(1, self.current_step),
            self.current_step / self.max_steps
        ])

        state = np.concatenate([
            self.current_embedding,
            self.current_hate_scores,
            self.current_target_features,
            self.user_history,
            platform_metrics
        ]).astype(np.float32)

        return state

    # Apply an action, compute reward, update state, and sample next comment.
    def step(self, action):
        self.current_step += 1
        self.total_actions += 1

        if self.current_target_toxicity is not None:
            toxicity_score = float(self.current_target_toxicity)
        else:
            toxicity_score = float(np.max(self.current_toxicity))

        reward = self._calculate_reward(action, toxicity_score)
        self._update_state(action, toxicity_score)

        self.current_idx = np.random.randint(0, len(self.embeddings))  # sample comment
        self.current_embedding = self.embeddings[self.current_idx]
        self.current_toxicity = self.labels[self.current_idx] if self.labels is not None else np.random.rand(6)
        if self.hate_scores is not None:
            self.current_hate_scores = self.hate_scores[self.current_idx]
        else:
            self.current_hate_scores = np.zeros(self.hate_score_dim, dtype=np.float32)
        if self.target_features is not None:
            self.current_target_features = self.target_features[self.current_idx]
        else:
            self.current_target_features = np.zeros(self.target_feature_dim, dtype=np.float32)
        if self.target_toxicity is not None:
            self.current_target_toxicity = float(self.target_toxicity[self.current_idx])
        else:
            self.current_target_toxicity = None

        done = self.current_step >= self.max_steps or self.platform_health <= 0.3  # episode end

        state = self._get_state()
        info = {
            'platform_health': self.platform_health,
            'user_satisfaction': self.user_satisfaction,
            'false_positive_rate': self.false_positive_count / max(1, self.total_actions),
            'action_taken': ModerationAction(action).name
        }

        return state, reward, done, False, info

    # Compute continuous alignment reward based on toxicity score and action severity.
    def _calculate_reward(self, action, toxicity_score):
        action_severity = action / 4.0
        alignment = 1.0 - abs(toxicity_score - action_severity)
        over_penalty = 0.5 * max(0.0, action_severity - toxicity_score)
        reward = alignment - over_penalty

        reward += 0.1 * self.platform_health  # health bonus

        return reward

    # Update user history and platform metrics based on action and toxicity.
    def _update_state(self, action, toxicity_score):
        if action == ModerationAction.WARN:
            self.user_history[1] += 1
        elif action == ModerationAction.REMOVE:
            self.user_history[2] += 1
        elif action == ModerationAction.TEMP_BAN:
            self.user_history[3] += 1
        elif action == ModerationAction.PERMA_BAN:
            self.user_history[4] += 1

        action_severity = action / 4.0
        if toxicity_score > 0.3 and action > ModerationAction.KEEP:
            self.platform_health += 0.01
        elif toxicity_score < 0.3 and action >= ModerationAction.TEMP_BAN:
            self.platform_health -= 0.02
            self.false_positive_count += 1

        if abs(toxicity_score - action_severity) < 0.3:
            self.user_satisfaction += 0.005
        else:
            self.user_satisfaction -= 0.01

        self.platform_health = np.clip(self.platform_health, 0.0, 1.0)  # clamp health
        self.user_satisfaction = np.clip(self.user_satisfaction, 0.0, 1.0)  # clamp satisfaction

    # No rendering implemented for this environment.
    def render(self):
        pass

    # No cleanup needed for this environment.
    def close(self):
        pass
