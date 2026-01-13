import axios from 'axios'; // Import HTTP client

const API_BASE_URL = 'http://localhost:8000'; // Define backend URL

export const moderateComment = async (comment) => { // Export moderation function
  const response = await axios.post(`${API_BASE_URL}/api/moderate`, { // Send POST request
    comment // Pass comment text
  });
  return response.data; // Return response data
};

export const getMetrics = async () => { // Export metrics function
  const response = await axios.get(`${API_BASE_URL}/api/metrics`); // Fetch metrics endpoint
  return response.data; // Return metrics data
};

export const getExamples = async () => { // Export examples function
  const response = await axios.get(`${API_BASE_URL}/api/examples`); // Fetch examples endpoint
  return response.data; // Return examples data
};
