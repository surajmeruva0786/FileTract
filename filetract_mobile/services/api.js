import axios from 'axios';

// Configure this to your deployed FileTract backend URL
// For local development: 'http://192.168.x.x:5000' (use your machine's LAN IP)
// For Render.com deployment: 'https://your-app.onrender.com'
const BASE_URL = 'https://your-filetract-backend.onrender.com';

const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 120000, // 2 minutes (patent pipeline can be slow)
  headers: { 'Content-Type': 'multipart/form-data' },
});

export const setBackendUrl = (url) => {
  apiClient.defaults.baseURL = url.replace(/\/$/, '');
};

export const getBackendUrl = () => apiClient.defaults.baseURL;

/**
 * Upload an image file to the FileTract backend.
 * @param {string} imageUri - Local URI from expo-image-picker
 * @param {string} filename - Display filename
 * @returns {Promise<{job_id: string, filename: string, status: string}>}
 */
export async function uploadImage(imageUri, filename = 'id_card.jpg') {
  const formData = new FormData();
  formData.append('file', {
    uri: imageUri,
    type: 'image/jpeg',
    name: filename,
  });

  const response = await apiClient.post('/api/upload', formData);
  return response.data;
}

/**
 * Trigger field extraction for an uploaded job.
 * @param {string} jobId
 * @param {string[]} fields - Array of field names to extract
 * @param {'standard'|'patent'} pipeline
 */
export async function extractFields(jobId, fields, pipeline = 'standard') {
  const response = await apiClient.post(
    '/api/extract',
    { job_id: jobId, fields, pipeline },
    { headers: { 'Content-Type': 'application/json' } }
  );
  return response.data;
}

/**
 * Poll job status until complete or error.
 * @param {string} jobId
 * @param {function} onProgress - Called with stage number (1-5)
 * @returns {Promise<object>} Final result
 */
export async function pollUntilComplete(jobId, onProgress, intervalMs = 2000) {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const statusRes = await apiClient.get(`/api/status/${jobId}`, {
          headers: { 'Content-Type': 'application/json' },
        });
        const { status, current_stage, error } = statusRes.data;

        if (onProgress && current_stage) onProgress(current_stage);

        if (status === 'complete') {
          const resultRes = await apiClient.get(`/api/result/${jobId}`, {
            headers: { 'Content-Type': 'application/json' },
          });
          resolve(resultRes.data);
        } else if (status === 'error') {
          reject(new Error(error || 'Processing failed'));
        } else {
          setTimeout(poll, intervalMs);
        }
      } catch (err) {
        reject(err);
      }
    };
    poll();
  });
}

/**
 * Full pipeline: upload → extract → poll → return results
 */
export async function processImage(imageUri, fields, pipeline = 'standard', onProgress) {
  const uploadResult = await uploadImage(imageUri);
  const jobId = uploadResult.job_id;

  await extractFields(jobId, fields, pipeline);

  const results = await pollUntilComplete(jobId, onProgress);
  return results;
}
