import React, { useState } from 'react'; // Import React hooks
import CommentInput from './components/CommentInput'; // Import CommentInput component
import ModerationResult from './components/ModerationResult'; // Import ModerationResult component
import { moderateComment } from './api'; // Import moderation API
import { Shield } from 'lucide-react'; // Import Shield icon

function App() { // Define App component
  const [loading, setLoading] = useState(false); // Initialize loading state
  const [result, setResult] = useState(null); // Initialize result state
  const [error, setError] = useState(null); // Initialize error state

  const handleModerate = async (comment) => { // Define moderation handler
    setLoading(true); // Set loading true
    setError(null); // Clear previous errors
    setResult(null); // Clear previous results

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

  return ( // Return JSX
    <div className="min-h-screen py-12 px-4"> {/* Main container */}
      <div className="container mx-auto flex flex-col items-center"> {/* Center content */}
        {/* Header */}
        <div className="text-center mb-12"> {/* Header section */}
          <div className="flex justify-center items-center gap-3 mb-4"> {/* Title row */}
            <Shield className="w-12 h-12 text-white" /> {/* Render Shield icon */}
            <h1 className="text-4xl font-bold text-white"> {/* Main title */}
              Content Moderation System {/* Title text */}
            </h1> {/* Close title */}
          </div> {/* Close title row */}
          <p className="text-purple-100 text-lg"> {/* Subtitle text */}
            AI-powered comment moderation using Deep Reinforcement Learning {/* Subtitle content */}
          </p> {/* Close subtitle */}
        </div> {/* Close header section */}

        {/* Main Content */}
        <CommentInput onSubmit={handleModerate} loading={loading} /> {/* Render input component */}

        {/* Error Message */}
        {error && ( // Conditionally render error
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg mt-6 max-w-3xl w-full"> {/* Error container */}
            <p className="font-semibold">Error:</p> {/* Error label */}
            <p>{error}</p> {/* Display error message */}
          </div> // Close error container
        )}

        {/* Result */}
        {result && <ModerationResult result={result} />} {/* Conditionally render result */}

        {/* Footer */}
        <div className="mt-12 text-center text-purple-100 text-sm"> {/* Footer container */}
          <p>Powered by DQN + DistilBERT + Detoxify</p> {/* Technology stack */}
          <p className="mt-1"> {/* Footer description */}
            Demonstrating responsible AI deployment with explainable RL {/* Footer text */}
          </p> {/* Close description */}
        </div> {/* Close footer */}
      </div> {/* Close center content */}
    </div> // Close main container
  );
}

export default App; // Export App component
