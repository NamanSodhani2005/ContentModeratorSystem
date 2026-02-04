import React, { useState } from 'react';
import { Send, Loader2 } from 'lucide-react';

const CommentInput = ({ onSubmit, loading }) => {
  const [comment, setComment] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (comment.trim()) {
      onSubmit(comment);
    }
  };

  return (
    <div className="card fade-in">
      <h2 className="card__title">Test Comment Moderation</h2>
      <p className="card__subtitle">
        Paste a sample comment to see the moderation action and rationale.
      </p>
      <form className="input-form" onSubmit={handleSubmit}>
        <textarea
          className="input-textarea"
          rows="4"
          placeholder="Type a comment to moderate..."
          value={comment}
          onChange={(e) => setComment(e.target.value)}
          disabled={loading}
        />
        <div className="input__meta">
          <span className="text-muted text-small">
            {comment.length} characters
          </span>
          <button
            type="submit"
            disabled={!comment.trim() || loading}
            className="button button--primary"
          >
            {loading ? (
              <>
                <Loader2 className="icon spinner" />
                Analyzing...
              </>
            ) : (
              <>
                <Send className="icon" />
                Moderate
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default CommentInput;
