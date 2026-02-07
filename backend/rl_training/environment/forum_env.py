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


    def __init__(
        self,
        embeddings,
        labels,
        target_features=None,
        target_toxicity=None,
        max_steps=500,
        toxic_sample_prob=0.35,
        toxicity_threshold=0.6
    ):
        super().__init__()

        self.embeddings = embeddings
        self.labels = labels
        self.target_features = target_features
        self.target_toxicity = target_toxicity
        self.max_steps = max_steps
        self.target_feature_dim = 4
        self.toxic_sample_prob = toxic_sample_prob
        self.toxicity_threshold = toxicity_threshold
        self.toxic_indices = None
        self.non_toxic_indices = None

        self.state_dim = 768 + self.target_feature_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(5)

        self.current_step = 0
        self.false_positive_count = 0
        self.total_actions = 0

        if self.labels is not None:
            label_max = self.labels.max(axis=1) if self.labels.ndim > 1 else self.labels
            toxic_mask = label_max >= self.toxicity_threshold
            self.toxic_indices = np.where(toxic_mask)[0]
            self.non_toxic_indices = np.where(~toxic_mask)[0]
            if len(self.toxic_indices) == 0 or len(self.non_toxic_indices) == 0:
                self.toxic_sample_prob = 0.0

        if self.target_toxicity is not None and len(self.target_toxicity) != len(self.embeddings):
            raise ValueError("target_toxicity must have the same length as embeddings")

        self.reset()


    def _sample_index(self):
        if self.toxic_indices is None or self.non_toxic_indices is None:
            return np.random.randint(0, len(self.embeddings))
        if np.random.rand() < self.toxic_sample_prob:
            return np.random.choice(self.toxic_indices)
        return np.random.choice(self.non_toxic_indices)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.false_positive_count = 0
        self.total_actions = 0

        self._load_comment(self._sample_index())
        return self._get_state(), {}


    def _load_comment(self, idx):
        self.current_idx = idx
        self.current_embedding = self.embeddings[idx]
        self.current_toxicity = self.labels[idx] if self.labels is not None else np.random.rand(6)
        if self.target_features is not None:
            self.current_target_features = self.target_features[idx]
        else:
            self.current_target_features = np.zeros(self.target_feature_dim, dtype=np.float32)
        if self.target_toxicity is not None:
            self.current_target_toxicity = float(self.target_toxicity[idx])
        else:
            self.current_target_toxicity = None


    def _get_state(self):
        return np.concatenate([
            self.current_embedding,
            self.current_target_features,
        ]).astype(np.float32)


    def step(self, action):
        self.current_step += 1
        self.total_actions += 1

        if self.current_target_toxicity is not None:
            toxicity_score = float(self.current_target_toxicity)
        else:
            toxicity_score = float(np.max(self.current_toxicity))

        reward = self._calculate_reward(action, toxicity_score)

        if toxicity_score < 0.3 and action >= ModerationAction.REMOVE:
            self.false_positive_count += 1

        self._load_comment(self._sample_index())

        done = self.current_step >= self.max_steps

        info = {
            'false_positive_rate': self.false_positive_count / max(1, self.total_actions),
            'action_taken': ModerationAction(action).name
        }

        return self._get_state(), reward, done, False, info


    def _calculate_reward(self, action, toxicity_score):
        action_severity = action / 4.0
        diff = toxicity_score - action_severity
        reward = 1.0 - 2.0 * diff * diff
        if diff < 0:
            reward -= 0.5 * abs(diff)
        return reward


    def render(self):
        pass


    def close(self):
        pass
