import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

export const moderateComment = async (comment) => {
  const response = await axios.post(`${API_BASE_URL}/api/moderate`, {
    comment
  });
  return response.data;
};

export const getMetrics = async () => {
  const response = await axios.get(`${API_BASE_URL}/api/metrics`);
  return response.data;
};

export const getExamples = async () => {
  const response = await axios.get(`${API_BASE_URL}/api/examples`);
  return response.data;
};

export const submitFeedback = async (comment, decision, feedback) => {
  const response = await axios.post(`${API_BASE_URL}/api/feedback`, {
    comment,
    decision,
    feedback
  });
  return response.data;
};
