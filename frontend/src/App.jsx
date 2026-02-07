import React, { useState } from 'react';
import CommentInput from './components/CommentInput';
import ModerationResult from './components/ModerationResult';
import { moderateComment, submitFeedback } from './api';
import { Shield } from 'lucide-react';

function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [lastComment, setLastComment] = useState('');

  const handleModerate = async (comment) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setLastComment(comment);

    try {
      const response = await moderateComment(comment);
      setResult(response);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to moderate comment. Make sure the backend is running.');
      console.error('Moderation error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (comment, decision, feedback) => {
    return submitFeedback(comment, decision, feedback);
  };

  return (
    <div className="app">
      <div className="app__shell">
        <header className="hero">
          <div className="hero__brand">
            <div className="hero__icon">
              <Shield className="icon icon--xl" />
            </div>
            <div>
              <h1 className="hero__title">Content Moderation System</h1>
              <p className="hero__subtitle">LLM for an automoderator with moderation actions</p>
            </div>
          </div>
          <p className="hero__lede">
            Refine moderation actions with realtime feedback using reinforcement learning.
          </p>
        </header>

        <CommentInput onSubmit={handleModerate} loading={loading} />

        {error && (
          <div className="error-banner">
            <p className="font-semibold">Error:</p>
            <p>{error}</p>
          </div>
        )}

        {result && (
          <ModerationResult
            result={result}
            comment={lastComment}
            onFeedback={handleFeedback}
          />
        )}

        <footer className="footer">
          <p className="footer__line">Powered by DQN + DistilBERT + Detoxify</p>
          <p className="footer__line">
            Demonstrating responsible AI deployment with explainable RL
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
