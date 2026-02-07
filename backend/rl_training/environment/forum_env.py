"""
Moderation gym environment.

Simulates a content moderation queue: at each step the agent sees a comment
(as an embedding + target features) and must choose a moderation action.
The reward encourages matching action severity to toxicity level, with an
extra penalty for over-moderation (false positives).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import IntEnum


class ModerationAction(IntEnum):
    KEEP = 0       # No action — leave the comment up
    WARN = 1       # Attach a warning label
    REMOVE = 2     # Remove the comment
    TEMP_BAN = 3   # Temporarily ban the user
    PERMA_BAN = 4  # Permanently ban the user


class ForumEnvironment(gym.Env):
    """Moderation environment.

    State: 772-dim vector = DistilBERT embedding (768) + target features (4).
    Actions: 5 discrete levels from KEEP (0) to PERMA_BAN (4).
    """

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

        # Pre-compute toxic/non-toxic index sets for stratified sampling.
        # This ensures the agent sees enough toxic examples during training
        # (toxic comments are typically <10% of the dataset).
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
        """Sample a comment index with oversampling of toxic examples.

        With probability toxic_sample_prob, pick a toxic comment; otherwise
        pick a non-toxic one. This balances the training signal so the agent
        learns to both act on toxic content and leave benign content alone.
        """
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
        """Load a comment's embedding, labels, and target features by index."""
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
        """Concatenate embedding + target features into a single state vector."""
        return np.concatenate([
            self.current_embedding,
            self.current_target_features,
        ]).astype(np.float32)

    def step(self, action):
        self.current_step += 1
        self.total_actions += 1

        # Use target-aware toxicity if available, else fall back to max label
        if self.current_target_toxicity is not None:
            toxicity_score = float(self.current_target_toxicity)
        else:
            toxicity_score = float(np.max(self.current_toxicity))

        reward = self._calculate_reward(action, toxicity_score)

        # Track false positives: harsh action (REMOVE+) on non-toxic content
        if toxicity_score < 0.3 and action >= ModerationAction.REMOVE:
            self.false_positive_count += 1

        # Move to next comment
        self._load_comment(self._sample_index())

        done = self.current_step >= self.max_steps

        info = {
            'false_positive_rate': self.false_positive_count / max(1, self.total_actions),
            'action_taken': ModerationAction(action).name
        }

        return self._get_state(), reward, done, False, info

    def _calculate_reward(self, action, toxicity_score):
        """Quadratic reward that encourages proportional moderation.

        Maps action to [0, 1] severity: KEEP=0, PERMA_BAN=1.
        Reward = 1 - 2*diff^2, with an additional -0.5*|diff| penalty when
        the agent over-moderates (action severity > toxicity). This asymmetry
        discourages false positives, which are more harmful to user trust
        than slightly under-moderating.
        """
        action_severity = action / 4.0
        diff = toxicity_score - action_severity
        reward = 1.0 - 2.0 * diff * diff
        # Extra penalty for over-moderation (diff < 0 means action > toxicity)
        if diff < 0:
            reward -= 0.5 * abs(diff)
        return reward

    def render(self):
        pass

    def close(self):
        pass


class VectorizedForumEnv:
    """Runs N moderation environments in parallel using vectorized numpy ops.

    Instead of Python-looping through one env at a time, this processes all
    N environments in a single numpy call per step. No multiprocessing needed —
    numpy vectorization is the speedup. This also means all N envs share the
    same underlying data arrays (no memory duplication).

    Auto-resets: when an env hits max_steps, it resets itself and the returned
    next_state is the first state of the new episode. The done flag is True for
    that step so DQN correctly ignores the next_state in the Bellman target.
    """

    def __init__(
        self,
        embeddings,
        labels,
        target_features=None,
        target_toxicity=None,
        max_steps=500,
        num_envs=16,
        toxic_sample_prob=0.35,
        toxicity_threshold=0.6
    ):
        self.embeddings = embeddings
        self.labels = labels
        self.target_features = target_features
        self.target_toxicity = target_toxicity
        self.max_steps = max_steps
        self.num_envs = num_envs
        self.target_feature_dim = 4
        self.toxic_sample_prob = toxic_sample_prob
        self.state_dim = 768 + self.target_feature_dim
        self.num_actions = 5

        # Pre-compute toxic/non-toxic index arrays for stratified sampling
        if labels is not None:
            label_max = labels.max(axis=1) if labels.ndim > 1 else labels
            toxic_mask = label_max >= toxicity_threshold
            self.toxic_indices = np.where(toxic_mask)[0]
            self.non_toxic_indices = np.where(~toxic_mask)[0]
            if len(self.toxic_indices) == 0 or len(self.non_toxic_indices) == 0:
                self.toxic_sample_prob = 0.0
        else:
            self.toxic_indices = None
            self.non_toxic_indices = None

        # Per-env counters
        self.current_steps = np.zeros(num_envs, dtype=np.int32)
        self.fp_counts = np.zeros(num_envs, dtype=np.int32)
        self.total_actions = np.zeros(num_envs, dtype=np.int32)
        self.episode_rewards = np.zeros(num_envs, dtype=np.float32)

        # Pre-allocated state output buffer: avoids np.concatenate every step
        self.state_buf = np.zeros((num_envs, self.state_dim), dtype=np.float32)
        self.current_indices = np.zeros(num_envs, dtype=np.int64)

    def _sample_indices(self, n):
        """Sample n comment indices with toxic oversampling (vectorized)."""
        if self.toxic_indices is None or self.non_toxic_indices is None:
            return np.random.randint(0, len(self.embeddings), size=n)
        is_toxic = np.random.rand(n) < self.toxic_sample_prob
        indices = np.empty(n, dtype=np.int64)
        n_toxic = is_toxic.sum()
        if n_toxic > 0:
            indices[is_toxic] = np.random.choice(self.toxic_indices, n_toxic)
        n_nontoxic = n - n_toxic
        if n_nontoxic > 0:
            indices[~is_toxic] = np.random.choice(self.non_toxic_indices, n_nontoxic)
        return indices

    def _load_comments(self, indices, mask=None):
        """Load embeddings + target features into state buffer.

        If mask is provided, only update those env slots.
        """
        if mask is None:
            self.current_indices[:] = indices
            self.state_buf[:, :768] = self.embeddings[indices]
            if self.target_features is not None:
                self.state_buf[:, 768:] = self.target_features[indices]
            else:
                self.state_buf[:, 768:] = 0
        else:
            self.current_indices[mask] = indices
            self.state_buf[mask, :768] = self.embeddings[indices]
            if self.target_features is not None:
                self.state_buf[mask, 768:] = self.target_features[indices]
            else:
                self.state_buf[mask, 768:] = 0

    def reset_all(self):
        """Reset all N envs and return initial states. Shape: (num_envs, state_dim)."""
        self.current_steps[:] = 0
        self.fp_counts[:] = 0
        self.total_actions[:] = 0
        self.episode_rewards[:] = 0
        self._load_comments(self._sample_indices(self.num_envs))
        return self.state_buf.copy()

    def step(self, actions):
        """Step all N envs simultaneously.

        Args:
            actions: (num_envs,) int array of moderation actions [0-4]
        Returns:
            next_states: (num_envs, state_dim) — post-reset state for done envs
            rewards: (num_envs,) float
            dones: (num_envs,) bool
            infos: dict with per-env episode stats (captured before auto-reset)
        """
        self.current_steps += 1
        self.total_actions += 1

        # Vectorized toxicity lookup
        if self.target_toxicity is not None:
            tox_scores = self.target_toxicity[self.current_indices].astype(np.float32)
        else:
            tox_scores = self.labels[self.current_indices].max(axis=1).astype(np.float32)

        # Vectorized reward: 1 - 2*diff^2, with over-moderation penalty
        action_severity = actions.astype(np.float32) / 4.0
        diff = tox_scores - action_severity
        rewards = 1.0 - 2.0 * diff * diff
        over_mod = diff < 0
        rewards[over_mod] -= 0.5 * np.abs(diff[over_mod])

        # Track false positives (harsh action on non-toxic content)
        fp_mask = (tox_scores < 0.3) & (actions >= 2)
        self.fp_counts += fp_mask.astype(np.int32)

        self.episode_rewards += rewards

        # Check which envs are done
        dones = self.current_steps >= self.max_steps

        # Capture per-env stats BEFORE auto-reset
        infos = {
            'episode_rewards': self.episode_rewards.copy(),
            'false_positive_rates': (self.fp_counts / np.maximum(1, self.total_actions)).copy(),
            'episode_lengths': self.current_steps.copy(),
        }

        # Load next comments for ALL envs (vectorized)
        next_indices = self._sample_indices(self.num_envs)
        self._load_comments(next_indices)

        # Auto-reset done envs (counters only — comments already loaded above)
        if dones.any():
            self.current_steps[dones] = 0
            self.fp_counts[dones] = 0
            self.total_actions[dones] = 0
            self.episode_rewards[dones] = 0

        return self.state_buf.copy(), rewards, dones, infos
