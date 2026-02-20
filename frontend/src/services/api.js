import axios from "axios";

// Use environment variable for API URL, fallback to localhost for development
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

// Log API URL for debugging
console.log("🔗 API Base URL:", API_BASE_URL);

// Health check
export const checkHealth = async () => {
  const response = await api.get("/health");
  return response.data;
};

// Client Management
export const createClient = async (clientId, description = "") => {
  const response = await api.post("/api/clients", {
    client_id: clientId,
    description,
  });
  return response.data;
};

export const listClients = async (skip = 0, limit = 100) => {
  const response = await api.get("/api/clients", {
    params: { skip, limit },
  });
  return response.data;
};

export const getClient = async (clientId) => {
  const response = await api.get(`/api/clients/${clientId}`);
  return response.data;
};

export const deleteClient = async (clientId) => {
  const response = await api.delete(`/api/clients/${clientId}`);
  return response.data;
};

// Document Management
export const uploadDocument = async (clientId, file) => {
  const formData = new FormData();
  formData.append("file", file);

  const response = await api.post(
    `/api/clients/${clientId}/documents`,
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    },
  );
  return response.data;
};

export const listDocuments = async (clientId) => {
  const response = await api.get(`/api/clients/${clientId}/documents`);
  return response.data;
};

export const clearDocuments = async (clientId) => {
  const response = await api.delete(`/api/clients/${clientId}/documents`);
  return response.data;
};

// Query & Chat
export const queryDocuments = async (
  clientId,
  question,
  includeSources = true,
  topK = 3,
) => {
  const response = await api.post(`/api/clients/${clientId}/query`, {
    question,
    include_sources: includeSources,
    top_k: topK,
  });
  return response.data;
};

export const chatWithDocuments = async (
  clientId,
  message,
  history = [],
  useRetrieval = true,
  topK = 3,
) => {
  const response = await api.post(`/api/clients/${clientId}/chat`, {
    message,
    history,
    use_retrieval: useRetrieval,
    top_k: topK,
  });
  return response.data;
};

export default api;
