# RAG Chatbot Frontend

React + Vite frontend for the multi-tenant RAG chatbot system.

## Features

- 🔧 Client Management - Create and manage multiple chatbot clients
- 📁 Document Upload - Upload PDF documents for each client
- 💬 Chat Interface - Interactive chat with document-augmented responses
- 🔍 Query Mode - Single-shot Q&A without conversation context
- 📚 Source Display - View document sources for AI responses

## Setup

1. Install dependencies:

```bash
npm install
```

2. Start development server:

```bash
npm run dev
```

3. Open http://localhost:3000

## Backend Connection

The frontend connects to the backend API at `http://localhost:8000`

Make sure the backend server is running before using the frontend.

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ClientManager.jsx     # Client CRUD operations
│   │   ├── DocumentUpload.jsx    # PDF upload interface
│   │   └── ChatInterface.jsx     # Chat & query interface
│   ├── services/
│   │   └── api.js                # API client
│   ├── App.jsx                   # Main application
│   ├── main.jsx                  # Entry point
│   └── index.css                 # Global styles
├── index.html
├── package.json
└── vite.config.js
```

## Usage

1. **Create a Client**: Enter a client ID and description, click "Create Client"
2. **Upload Documents**: Select the client, upload PDF files
3. **Start Chatting**: Ask questions in the chat interface
4. **Switch Modes**: Toggle between Chat (conversational) and Query (one-shot) modes
5. **View Sources**: Enable "Show Sources" to see document references

## Development

- Built with React 18 + Vite
- Axios for API calls
- Responsive design with CSS
- No external UI libraries for lightweight bundle
