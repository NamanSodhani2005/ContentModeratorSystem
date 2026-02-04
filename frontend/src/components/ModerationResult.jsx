import React, { useEffect, useState } from 'react';
import { Shield, AlertTriangle, X, Ban, XCircle, CheckCircle } from 'lucide-react';

const ModerationResult = ({ result, comment, onFeedback }) => {
  if (!result) return null;

  const [feedbackState, setFeedbackState] = useState({
    status: 'idle',
    choice: null,
    error: null,
    message: null,
    updated: false
  });

  useEffect(() => {
    setFeedbackState({ status: 'idle', choice: null, error: null, message: null, updated: false });
  }, [result?.decision, comment]);

  const getActionIcon = (decision) => {
    const icons = {
      keep: <CheckCircle className="icon icon--lg" />,
      warn: <AlertTriangle className="icon icon--lg" />,
      remove: <X className="icon icon--lg" />,
      temp_ban: <Ban className="icon icon--lg" />,
      perma_ban: <XCircle className="icon icon--lg" />
    };
    return icons[decision] || <Shield className="icon icon--lg" />;
  };

  const toxicityCategories = [
    { key: 'toxicity', label: 'Toxicity' },
    { key: 'severe_toxicity', label: 'Severe Toxicity' },
    { key: 'obscene', label: 'Obscene' },
    { key: 'threat', label: 'Threat' },
    { key: 'insult', label: 'Insult' },
    { key: 'identity_attack', label: 'Identity Attack' }
  ];

  const feedbackOptions = [
    { key: 'too_soft', label: 'Too lenient' },
    { key: 'good', label: 'Just right' },
    { key: 'too_harsh', label: 'Too harsh' }
  ];

  const canSendFeedback = Boolean(onFeedback && comment && result?.decision);
  const isSubmitting = feedbackState.status === 'submitting';

  const handleFeedback = async (feedbackKey) => {
    if (!canSendFeedback || isSubmitting) {
      return;
    }

    setFeedbackState({ status: 'submitting', choice: feedbackKey, error: null, message: null, updated: false });
    try {
      const response = await onFeedback(comment, result.decision, feedbackKey);
      const updated = Boolean(response?.updated);
      const message = updated
        ? `Model updated (${response.update_steps || 0} steps).`
        : 'Feedback saved. Model will update after more feedback.';
      setFeedbackState({ status: 'success', choice: feedbackKey, error: null, message, updated });
    } catch (err) {
      const message = err.response?.data?.detail || 'Failed to send feedback.';
      setFeedbackState({ status: 'error', choice: feedbackKey, error: message, message: null, updated: false });
    }
  };

  return (
    <div className="card result-card fade-in">
      <div className="result__header">
        <div className="decision-badge" data-action={result.decision}>
          {getActionIcon(result.decision)}
        </div>
        <div className="result__meta">
          <h3 className="result__title">{result.decision.replace('_', ' ')}</h3>
          <p className="result__confidence">
            Confidence: {(result.confidence * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      <div className="panel">
        <h4 className="panel__title">Reasoning</h4>
        <p className="text-muted">{result.reasoning}</p>
      </div>

      <div>
        <h4 className="panel__title">Toxicity Analysis</h4>
        <div className="toxicity">
          {toxicityCategories.map(({ key, label }) => {
            const score = result.toxicity_breakdown[key] || 0;
            const percentage = (score * 100).toFixed(1);
            const barColor = score > 0.7
              ? '#111111'
              : score > 0.5
                ? '#6b7280'
                : '#cbd5e1';

            return (
              <div key={key} className="toxicity__row">
                <div className="toxicity__label">
                  <span>{label}</span>
                  <span>{percentage}%</span>
                </div>
                <div className="progress">
                  <div
                    className="progress__fill"
                    style={{ width: `${percentage}%`, backgroundColor: barColor }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div>
        <h4 className="panel__title">Was this decision fair?</h4>
        <div className="feedback__buttons">
          {feedbackOptions.map((option) => {
            const isSelected = feedbackState.choice === option.key;
            const isDisabled = !canSendFeedback || isSubmitting;
            const buttonClass = [
              'feedback__button',
              isSelected ? 'is-selected' : '',
              isDisabled ? 'is-disabled' : ''
            ].filter(Boolean).join(' ');

            return (
              <button
                key={option.key}
                type="button"
                onClick={() => handleFeedback(option.key)}
                disabled={isDisabled}
                className={buttonClass}
              >
                {option.label}
              </button>
            );
          })}
        </div>
        <p className="feedback__note">Your feedback updates the model online.</p>
        {feedbackState.status === 'success' && feedbackState.message && (
          <p className={`feedback__message ${feedbackState.updated ? 'feedback__message--ok' : 'feedback__message--info'}`}>
            {feedbackState.message}
          </p>
        )}
        {feedbackState.status === 'error' && (
          <p className="feedback__message feedback__message--error">{feedbackState.error}</p>
        )}
      </div>

      <div>
        <h4 className="panel__title">Alternative Actions (by Q-value)</h4>
        <div className="alt-actions">
          {result.alternative_actions.slice(0, 3).map((alt) => {
            const isChosen = alt.action === result.decision;
            return (
              <div
                key={alt.action}
                className={`alt-action ${isChosen ? 'is-selected' : ''}`}
              >
                <div className="alt-action__left">
                  {isChosen && <span className="badge">CHOSEN</span>}
                  <span>{alt.action.replace('_', ' ')}</span>
                </div>
                <div className="alt-action__meta">
                  <span>Q: {alt.q_value.toFixed(3)}</span>
                  <span>P: {(alt.probability * 100).toFixed(1)}%</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default ModerationResult;
