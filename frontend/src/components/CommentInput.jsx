import React, { useState } from 'react'; // Import React hooks
import { Send, Loader2 } from 'lucide-react'; // Import icons

const CommentInput = ({ onSubmit, loading }) => { // Define CommentInput component
  const [comment, setComment] = useState(''); // Initialize comment state

  const handleSubmit = (e) => { // Define submit handler
    e.preventDefault(); // Prevent default form submit
    if (comment.trim()) { // Check comment not empty
      onSubmit(comment); // Call onSubmit prop
    }
  };

  return ( // Return JSX
    <div className="card fade-in"> {/* Main container */}
      <h2 className="card__title"> {/* Section title */}
        Test Comment Moderation {/* Title text */}
      </h2> {/* Close title */}
      <p className="card__subtitle"> {/* Subtitle */}
        Paste a sample comment to see the moderation action and rationale. {/* Subtitle text */}
      </p> {/* Close subtitle */}
      <form className="input-form" onSubmit={handleSubmit}> {/* Form element */}
        <textarea // Textarea input element
          className="input-textarea" // Textarea styles
          rows="4" // Set row count
          placeholder="Type a comment to moderate..." // Placeholder text
          value={comment} // Bind to state
          onChange={(e) => setComment(e.target.value)} // Update state on change
          disabled={loading} // Disable when loading
        />
        <div className="input__meta"> {/* Bottom row */}
          <span className="text-muted text-small"> {/* Character count label */}
            {comment.length} characters {/* Display character count */}
          </span> {/* Close count label */}
          <button // Submit button element
            type="submit" // Button type submit
            disabled={!comment.trim() || loading} // Disable if empty/loading
            className="button button--primary" // Button styles
          >
            {loading ? ( // Conditional render loading
              <> {/* Fragment for loading */}
                <Loader2 className="icon spinner" /> {/* Animated spinner icon */}
                Analyzing... {/* Loading text */}
              </> // Close loading fragment
            ) : ( // Conditional render normal
              <> {/* Fragment for normal */}
                <Send className="icon" /> {/* Send icon */}
                Moderate {/* Button text */}
              </> // Close normal fragment
            )}
          </button> {/* Close button */}
        </div> {/* Close bottom row */}
      </form> {/* Close form */}
    </div> // Close main container
  );
};

export default CommentInput; // Export CommentInput component
