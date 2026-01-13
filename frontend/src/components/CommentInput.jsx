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
    <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-3xl"> {/* Main container */}
      <h2 className="text-2xl font-bold text-gray-800 mb-4"> {/* Section title */}
        Test Comment Moderation {/* Title text */}
      </h2> {/* Close title */}
      <form onSubmit={handleSubmit}> {/* Form element */}
        <textarea // Textarea input element
          className="w-full p-4 border-2 border-gray-300 rounded-lg focus:border-purple-500 focus:outline-none transition-colors resize-none" // Textarea styles
          rows="4" // Set row count
          placeholder="Type a comment to moderate..." // Placeholder text
          value={comment} // Bind to state
          onChange={(e) => setComment(e.target.value)} // Update state on change
          disabled={loading} // Disable when loading
        />
        <div className="flex justify-between items-center mt-4"> {/* Bottom row */}
          <span className="text-sm text-gray-500"> {/* Character count label */}
            {comment.length} characters {/* Display character count */}
          </span> {/* Close count label */}
          <button // Submit button element
            type="submit" // Button type submit
            disabled={!comment.trim() || loading} // Disable if empty/loading
            className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white font-semibold py-2 px-6 rounded-lg flex items-center gap-2 transition-colors" // Button styles
          >
            {loading ? ( // Conditional render loading
              <> {/* Fragment for loading */}
                <Loader2 className="w-5 h-5 animate-spin" /> {/* Animated spinner icon */}
                Analyzing... {/* Loading text */}
              </> // Close loading fragment
            ) : ( // Conditional render normal
              <> {/* Fragment for normal */}
                <Send className="w-5 h-5" /> {/* Send icon */}
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
