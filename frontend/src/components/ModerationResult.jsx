import React, { useEffect, useState } from 'react'; // Import React hooks
import { Shield, AlertTriangle, X, Ban, XCircle, CheckCircle } from 'lucide-react'; // Import all icons

const ModerationResult = ({ result, comment, onFeedback }) => { // Define ModerationResult component
  if (!result) return null; // Return null if no result

  const [feedbackState, setFeedbackState] = useState({ // Track feedback state
    status: 'idle',
    choice: null,
    error: null,
    message: null,
    updated: false // Update flag
  });

  useEffect(() => { // Reset feedback on change
    setFeedbackState({ status: 'idle', choice: null, error: null, message: null, updated: false });
  }, [result?.decision, comment]);

  const getActionIcon = (decision) => { // Define icon getter function
    const icons = { // Define icon map
      keep: <CheckCircle className="icon icon--lg" />, // Keep action icon
      warn: <AlertTriangle className="icon icon--lg" />, // Warn action icon
      remove: <X className="icon icon--lg" />, // Remove action icon
      temp_ban: <Ban className="icon icon--lg" />, // Temp ban icon
      perma_ban: <XCircle className="icon icon--lg" /> // Perma ban icon
    };
    return icons[decision] || <Shield className="icon icon--lg" />; // Return icon or default
  };

  const toxicityCategories = [ // Define categories array
    { key: 'toxicity', label: 'Toxicity' }, // Toxicity category
    { key: 'severe_toxicity', label: 'Severe Toxicity' }, // Severe toxicity category
    { key: 'obscene', label: 'Obscene' }, // Obscene category
    { key: 'threat', label: 'Threat' }, // Threat category
    { key: 'insult', label: 'Insult' }, // Insult category
    { key: 'identity_attack', label: 'Identity Attack' } // Identity attack category
  ];

  const feedbackOptions = [ // Feedback options
    { key: 'too_soft', label: 'Too lenient' },
    { key: 'good', label: 'Just right' },
    { key: 'too_harsh', label: 'Too harsh' }
  ];

  const canSendFeedback = Boolean(onFeedback && comment && result?.decision); // Validate feedback inputs
  const isSubmitting = feedbackState.status === 'submitting'; // Submission state

  const handleFeedback = async (feedbackKey) => { // Handle feedback submit
    if (!canSendFeedback || isSubmitting) { // Guard invalid state
      return;
    }

    setFeedbackState({ status: 'submitting', choice: feedbackKey, error: null, message: null, updated: false });
    try {
      const response = await onFeedback(comment, result.decision, feedbackKey);
      const updated = Boolean(response?.updated);
      // Build feedback message
      const message = updated
        ? `Model updated (${response.update_steps || 0} steps).`
        : 'Feedback saved. Model will update after more feedback.';
      setFeedbackState({ status: 'success', choice: feedbackKey, error: null, message, updated });
    } catch (err) {
      const message = err.response?.data?.detail || 'Failed to send feedback.';
      setFeedbackState({ status: 'error', choice: feedbackKey, error: message, message: null, updated: false });
    }
  };

  return ( // Return JSX
    <div className="card result-card fade-in"> {/* Main result container */}
      {/* Decision Header */}
      <div className="result__header"> {/* Header row */}
        <div className="decision-badge" data-action={result.decision}> {/* Icon container */}
          {getActionIcon(result.decision)} {/* Render decision icon */}
        </div> {/* Close icon container */}
        <div className="result__meta"> {/* Text container */}
          <h3 className="result__title"> {/* Decision title */}
            {result.decision.replace('_', ' ')} {/* Display decision text */}
          </h3> {/* Close title */}
          <p className="result__confidence"> {/* Confidence text */}
            Confidence: {(result.confidence * 100).toFixed(1)}% {/* Display confidence percentage */}
          </p> {/* Close confidence */}
        </div> {/* Close text container */}
      </div> {/* Close header row */}

      {/* Reasoning */}
      <div className="panel"> {/* Reasoning container */}
        <h4 className="panel__title">Reasoning</h4> {/* Reasoning title */}
        <p className="text-muted">{result.reasoning}</p> {/* Display reasoning text */}
      </div> {/* Close reasoning container */}

      {/* Toxicity Breakdown */}
      <div> {/* Toxicity section */}
        <h4 className="panel__title">Toxicity Analysis</h4> {/* Section title */}
        <div className="toxicity"> {/* Categories container */}
          {toxicityCategories.map(({ key, label }) => { // Map through categories
            const score = result.toxicity_breakdown[key] || 0; // Get score or zero
            const percentage = (score * 100).toFixed(1); // Calculate percentage
            const barColor = score > 0.7
              ? '#111111'
              : score > 0.5
                ? '#6b7280'
                : '#cbd5e1';

            return ( // Return category element
              <div key={key} className="toxicity__row"> {/* Category container */}
                <div className="toxicity__label"> {/* Label row */}
                  <span>{label}</span> {/* Category label */}
                  <span>{percentage}%</span> {/* Score percentage */}
                </div> {/* Close label row */}
                <div className="progress"> {/* Progress bar background */}
                  <div // Progress bar fill
                    className="progress__fill" // Bar fill styles
                    style={{ width: `${percentage}%`, backgroundColor: barColor }} // Set bar width
                  />
                </div> {/* Close progress background */}
              </div> // Close category container
            );
          })}
        </div> {/* Close categories container */}
      </div> {/* Close toxicity section */}

      {/* Feedback */}
      <div> {/* Feedback section */}
        <h4 className="panel__title">Was this decision fair?</h4> {/* Section title */}
        <div className="feedback__buttons"> {/* Button row */}
          {feedbackOptions.map((option) => { // Map feedback options
            const isSelected = feedbackState.choice === option.key; // Selected state
            const isDisabled = !canSendFeedback || isSubmitting;
            const buttonClass = [
              'feedback__button',
              isSelected ? 'is-selected' : '',
              isDisabled ? 'is-disabled' : ''
            ].filter(Boolean).join(' ');

            return ( // Return feedback button
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
        </div> {/* Close button row */}
        <p className="feedback__note">Your feedback updates the model online.</p> {/* Helper text */}
        {feedbackState.status === 'success' && feedbackState.message && ( // Success message
          <p className={`feedback__message ${feedbackState.updated ? 'feedback__message--ok' : 'feedback__message--info'}`}>
            {feedbackState.message}
          </p>
        )}
        {feedbackState.status === 'error' && ( // Error message
          <p className="feedback__message feedback__message--error">{feedbackState.error}</p>
        )}
      </div> {/* Close feedback section */}

      {/* Alternative Actions */}
      <div> {/* Alternatives section */}
        <h4 className="panel__title">Alternative Actions (by Q-value)</h4> {/* Section title */}
        <div className="alt-actions"> {/* Actions container */}
          {result.alternative_actions.slice(0, 3).map((alt) => { // Map top 3 alternatives
            const isChosen = alt.action === result.decision; // Check chosen action
            return ( // Return action row
              <div // Action row
                key={alt.action} // Unique key
                className={`alt-action ${isChosen ? 'is-selected' : ''}`} // Row base styles
              >
                <div className="alt-action__left"> {/* Left side */}
                  {isChosen && <span className="badge">CHOSEN</span>} {/* Show chosen badge */}
                  <span> {/* Action name */}
                    {alt.action.replace('_', ' ')} {/* Display action text */}
                  </span> {/* Close action name */}
                </div> {/* Close left side */}
                <div className="alt-action__meta"> {/* Right side */}
                  <span> {/* Q-value label */}
                    Q: {alt.q_value.toFixed(3)} {/* Display Q-value */}
                  </span> {/* Close Q-value */}
                  <span> {/* Probability label */}
                    P: {(alt.probability * 100).toFixed(1)}% {/* Display probability */}
                  </span> {/* Close probability */}
                </div> {/* Close right side */}
              </div> // Close action row
            );
          })}
        </div> {/* Close actions container */}
      </div> {/* Close alternatives section */}
    </div> // Close main container
  );
};

export default ModerationResult; // Export ModerationResult component
