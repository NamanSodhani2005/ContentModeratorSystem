import React from 'react'; // Import React library
import { Shield, AlertTriangle, X, Ban, XCircle, CheckCircle } from 'lucide-react'; // Import all icons

const ModerationResult = ({ result }) => { // Define ModerationResult component
  if (!result) return null; // Return null if no result

  const getActionColor = (decision) => { // Define color getter function
    const colors = { // Define color map
      keep: 'bg-green-100 text-green-800 border-green-300', // Keep action colors
      warn: 'bg-yellow-100 text-yellow-800 border-yellow-300', // Warn action colors
      remove: 'bg-orange-100 text-orange-800 border-orange-300', // Remove action colors
      temp_ban: 'bg-red-100 text-red-800 border-red-300', // Temp ban colors
      perma_ban: 'bg-red-200 text-red-900 border-red-400' // Perma ban colors
    };
    return colors[decision] || 'bg-gray-100 text-gray-800 border-gray-300'; // Return color or default
  };

  const getActionIcon = (decision) => { // Define icon getter function
    const icons = { // Define icon map
      keep: <CheckCircle className="w-6 h-6" />, // Keep action icon
      warn: <AlertTriangle className="w-6 h-6" />, // Warn action icon
      remove: <X className="w-6 h-6" />, // Remove action icon
      temp_ban: <Ban className="w-6 h-6" />, // Temp ban icon
      perma_ban: <XCircle className="w-6 h-6" /> // Perma ban icon
    };
    return icons[decision] || <Shield className="w-6 h-6" />; // Return icon or default
  };

  const toxicityCategories = [ // Define categories array
    { key: 'toxicity', label: 'Toxicity' }, // Toxicity category
    { key: 'severe_toxicity', label: 'Severe Toxicity' }, // Severe toxicity category
    { key: 'obscene', label: 'Obscene' }, // Obscene category
    { key: 'threat', label: 'Threat' }, // Threat category
    { key: 'insult', label: 'Insult' }, // Insult category
    { key: 'identity_attack', label: 'Identity Attack' } // Identity attack category
  ];

  return ( // Return JSX
    <div className="bg-white rounded-lg shadow-xl p-6 w-full max-w-3xl mt-6 animate-fade-in"> {/* Main result container */}
      {/* Decision Header */}
      <div className="flex items-center gap-4 mb-6"> {/* Header row */}
        <div className={`p-3 rounded-full ${getActionColor(result.decision)}`}> {/* Icon container */}
          {getActionIcon(result.decision)} {/* Render decision icon */}
        </div> {/* Close icon container */}
        <div> {/* Text container */}
          <h3 className="text-2xl font-bold text-gray-800 capitalize"> {/* Decision title */}
            {result.decision.replace('_', ' ')} {/* Display decision text */}
          </h3> {/* Close title */}
          <p className="text-sm text-gray-600"> {/* Confidence text */}
            Confidence: {(result.confidence * 100).toFixed(1)}% {/* Display confidence percentage */}
          </p> {/* Close confidence */}
        </div> {/* Close text container */}
      </div> {/* Close header row */}

      {/* Reasoning */}
      <div className="mb-6 p-4 bg-gray-50 rounded-lg border border-gray-200"> {/* Reasoning container */}
        <h4 className="font-semibold text-gray-700 mb-2">Reasoning</h4> {/* Reasoning title */}
        <p className="text-gray-600">{result.reasoning}</p> {/* Display reasoning text */}
      </div> {/* Close reasoning container */}

      {/* Toxicity Breakdown */}
      <div className="mb-6"> {/* Toxicity section */}
        <h4 className="font-semibold text-gray-700 mb-3">Toxicity Analysis</h4> {/* Section title */}
        <div className="space-y-3"> {/* Categories container */}
          {toxicityCategories.map(({ key, label }) => { // Map through categories
            const score = result.toxicity_breakdown[key] || 0; // Get score or zero
            const percentage = (score * 100).toFixed(1); // Calculate percentage
            const barColor = score > 0.7 ? 'bg-red-500' : score > 0.5 ? 'bg-orange-500' : 'bg-green-500'; // Determine bar color

            return ( // Return category element
              <div key={key}> {/* Category container */}
                <div className="flex justify-between text-sm mb-1"> {/* Label row */}
                  <span className="text-gray-700">{label}</span> {/* Category label */}
                  <span className="text-gray-600 font-medium">{percentage}%</span> {/* Score percentage */}
                </div> {/* Close label row */}
                <div className="w-full bg-gray-200 rounded-full h-2"> {/* Progress bar background */}
                  <div // Progress bar fill
                    className={`${barColor} h-2 rounded-full transition-all duration-500`} // Bar fill styles
                    style={{ width: `${percentage}%` }} // Set bar width
                  />
                </div> {/* Close progress background */}
              </div> // Close category container
            );
          })}
        </div> {/* Close categories container */}
      </div> {/* Close toxicity section */}

      {/* Alternative Actions */}
      <div> {/* Alternatives section */}
        <h4 className="font-semibold text-gray-700 mb-3">Alternative Actions (by Q-value)</h4> {/* Section title */}
        <div className="space-y-2"> {/* Actions container */}
          {result.alternative_actions.slice(0, 3).map((alt) => { // Map top 3 alternatives
            const isChosen = alt.action === result.decision; // Check chosen action
            return ( // Return action row
              <div // Action row
                key={alt.action} // Unique key
                className={`flex justify-between items-center p-3 rounded-lg border ${ // Row base styles
                  isChosen ? 'bg-purple-50 border-purple-300' : 'bg-gray-50 border-gray-200' // Conditional styling
                }`}
              >
                <div className="flex items-center gap-2"> {/* Left side */}
                  {isChosen && <span className="text-xs font-bold text-purple-700">CHOSEN</span>} {/* Show chosen badge */}
                  <span className="font-medium text-gray-800 capitalize"> {/* Action name */}
                    {alt.action.replace('_', ' ')} {/* Display action text */}
                  </span> {/* Close action name */}
                </div> {/* Close left side */}
                <div className="flex items-center gap-4"> {/* Right side */}
                  <span className="text-sm text-gray-600"> {/* Q-value label */}
                    Q: {alt.q_value.toFixed(3)} {/* Display Q-value */}
                  </span> {/* Close Q-value */}
                  <span className="text-sm text-gray-600"> {/* Probability label */}
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
