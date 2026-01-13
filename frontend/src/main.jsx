import React from 'react' // Import React library
import ReactDOM from 'react-dom/client' // Import ReactDOM client
import App from './App.jsx' // Import App component
import './index.css' // Import global styles

ReactDOM.createRoot(document.getElementById('root')).render( // Create root and render
  <React.StrictMode> {/* Enable strict mode */}
    <App /> {/* Render App component */}
  </React.StrictMode>, // Close StrictMode
)
