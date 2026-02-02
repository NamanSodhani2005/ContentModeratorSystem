import React, { useState } from 'react'; // Import React hooks
import CommentInput from './components/CommentInput'; // Import CommentInput component
import ModerationResult from './components/ModerationResult'; // Import ModerationResult component
import { moderateComment, submitFeedback } from './api'; // Import moderation API
import { Shield } from 'lucide-react'; // Import Shield icon

function App() { // Define App component
  const [loading, setLoading] = useState(false); // Initialize loading state
  const [result, setResult] = useState(null); // Initialize result state
  const [error, setError] = useState(null); // Initialize error state
  const [lastComment, setLastComment] = useState(''); // Store last moderated comment

  const handleModerate = async (comment) => { // Define moderation handler
    setLoading(true); // Set loading true
    setError(null); // Clear previous errors
    setResult(null); // Clear previous results
    setLastComment(comment); // Save comment for feedback

    try { // Try API call
      const response = await moderateComment(comment); // Call moderation API
      setResult(response); // Store response data
    } catch (err) { // Catch API errors
      setError(err.response?.data?.detail || 'Failed to moderate comment. Make sure the backend is running.'); // Set error message
      console.error('Moderation error:', err); // Log error to console
    } finally { // Execute after try/catch
      setLoading(false); // Set loading false
    }
  };

  const handleFeedback = async (comment, decision, feedback) => { // Define feedback handler
    return submitFeedback(comment, decision, feedback); // Send feedback to backend
  };

  return ( // Return JSX
    <div className="app"> {/* Main container */}
      <div className="app__shell"> {/* Center content */}
        {/* Header */}
        <header className="hero"> {/* Header section */}
          <div className="hero__brand"> {/* Title row */}
            <div className="hero__icon"> {/* Icon container */}
              <Shield className="icon icon--xl" /> {/* Render Shield icon */}
            </div> {/* Close icon container */}
            <div> {/* Title copy */}
              <h1 className="hero__title"> {/* Main title */}
                Content Moderation System {/* Title text */}
              </h1> {/* Close title */}
              <p className="hero__subtitle"> {/* Subtitle text */}
                DQN policy over contextual toxicity signals {/* Subtitle content */}
              </p> {/* Close subtitle */}
            </div> {/* Close title copy */}
          </div> {/* Close title row */}
          <p className="hero__lede"> {/* Supporting line */}
            Evaluate, explain, and refine moderation actions with realtime feedback. {/* Supporting text */}
          </p> {/* Close supporting line */}
        </header> {/* Close header section */}

        {/* Main Content */}
        <CommentInput onSubmit={handleModerate} loading={loading} /> {/* Render input component */}

        {/* Error Message */}
        {error && ( // Conditionally render error
          <div className="error-banner"> {/* Error container */}
            <p className="font-semibold">Error:</p> {/* Error label */}
            <p>{error}</p> {/* Display error message */}
          </div> // Close error container
        )}

        {/* Result */}
        {result && ( // Conditionally render result
          <ModerationResult // Render result component
            result={result} // Pass result data
            comment={lastComment} // Pass original comment
            onFeedback={handleFeedback} // Pass feedback handler
          />
        )}

        {/* Footer */}
        <footer className="footer"> {/* Footer container */}
          <p className="footer__line">Powered by DQN + DistilBERT + Detoxify</p> {/* Technology stack */}
          <p className="footer__line"> {/* Footer description */}
            Demonstrating responsible AI deployment with explainable RL {/* Footer text */}
          </p> {/* Close description */}
        </footer> {/* Close footer */}
      </div> {/* Close center content */}
    </div> // Close main container
  );
}

export default App; // Export App component
