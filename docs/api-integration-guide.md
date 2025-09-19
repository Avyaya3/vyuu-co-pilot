# API Integration Guide for Chatbot UI

This guide explains how to integrate your LangGraph-based Vyuu Copilot v2 system with a chatbot UI using the FastAPI endpoints.

## 🚀 Quick Start

### 1. Start the API Server

```bash
# From the project root
python start_api.py
```

The API will be available at `http://localhost:8000` with automatic documentation at `http://localhost:8000/docs`.

### 2. Basic Chat Integration

Here's a simple JavaScript example for integrating with your chatbot UI:

```javascript
class VyuuCopilotClient {
    constructor(baseUrl = 'http://localhost:8000', authToken = null) {
        this.baseUrl = baseUrl;
        this.authToken = authToken;
        this.sessionId = null;
    }

    async sendMessage(message, conversationHistory = []) {
        const headers = {
            'Content-Type': 'application/json',
        };

        if (this.authToken) {
            headers['Authorization'] = `Bearer ${this.authToken}`;
        }

        const response = await fetch(`${this.baseUrl}/chat`, {
            method: 'POST',
            headers,
            body: JSON.stringify({
                message: message,
                session_id: this.sessionId,
                conversation_history: conversationHistory,
            }),
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        
        // Store session ID for conversation continuity
        if (data.session_id) {
            this.sessionId = data.session_id;
        }

        return data;
    }

    async getSessionHistory() {
        if (!this.sessionId) return [];

        const headers = {};
        if (this.authToken) {
            headers['Authorization'] = `Bearer ${this.authToken}`;
        }

        const response = await fetch(`${this.baseUrl}/sessions/${this.sessionId}/history`, {
            headers,
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }

        return await response.json();
    }

    async clearSession() {
        if (!this.sessionId) return;

        const headers = {};
        if (this.authToken) {
            headers['Authorization'] = `Bearer ${this.authToken}`;
        }

        await fetch(`${this.baseUrl}/sessions/${this.sessionId}`, {
            method: 'DELETE',
            headers,
        });

        this.sessionId = null;
    }
}

// Usage example
const copilot = new VyuuCopilotClient();

// Send a message
async function sendMessage() {
    const message = document.getElementById('messageInput').value;
    const response = await copilot.sendMessage(message);
    
    // Display response
    document.getElementById('response').textContent = response.response;
    
    // Update conversation history
    updateConversationHistory(response.conversation_history);
}

// Update UI with conversation history
function updateConversationHistory(history) {
    const chatContainer = document.getElementById('chatContainer');
    chatContainer.innerHTML = '';
    
    history.forEach(msg => {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${msg.role}`;
        messageDiv.innerHTML = `
            <div class="content">${msg.content}</div>
            <div class="timestamp">${msg.timestamp}</div>
        `;
        chatContainer.appendChild(messageDiv);
    });
}
```

## 📡 API Endpoints

### 1. Chat Endpoint

**POST** `/chat`

Main endpoint for sending messages to the copilot.

**Request Body:**
```json
{
    "message": "I need help with my budget",
    "session_id": "optional-session-id",
    "conversation_history": [
        {
            "role": "user",
            "content": "Hello",
            "timestamp": "2024-01-01T10:00:00Z"
        },
        {
            "role": "assistant", 
            "content": "Hi! How can I help you today?",
            "timestamp": "2024-01-01T10:00:01Z"
        }
    ],
    "user_id": "optional-user-id"
}
```

**Response:**
```json
{
    "response": "I'd be happy to help you with your budget! Let me ask a few questions to better understand your financial situation...",
    "session_id": "generated-session-id",
    "status": "success",
    "conversation_history": [
        {
            "role": "user",
            "content": "I need help with my budget",
            "timestamp": "2024-01-01T10:01:00Z"
        },
        {
            "role": "assistant",
            "content": "I'd be happy to help you with your budget! Let me ask a few questions...",
            "timestamp": "2024-01-01T10:01:01Z"
        }
    ],
    "metadata": {
        "intent": "budget_help",
        "confidence": 0.95,
        "processing_time_ms": 1250.5
    },
    "processing_time_ms": 1250.5
}
```

### 2. Simple Chat Endpoint

**POST** `/chat/simple`

Simplified endpoint that accepts just a message string.

**Request Body:**
```json
{
    "message": "What's my current savings balance?"
}
```

**Response:**
```json
{
    "response": "Your current savings balance is $5,250.00",
    "session_id": "generated-session-id",
    "status": "success"
}
```

### 3. Session History

**GET** `/sessions/{session_id}/history`

Retrieve conversation history for a session.

**Response:**
```json
[
    {
        "role": "user",
        "content": "Hello",
        "timestamp": "2024-01-01T10:00:00Z"
    },
    {
        "role": "assistant",
        "content": "Hi! How can I help you today?",
        "timestamp": "2024-01-01T10:00:01Z"
    }
]
```

### 4. Clear Session

**DELETE** `/sessions/{session_id}`

Clear conversation history for a session.

**Response:**
```json
{
    "message": "Session cleared successfully",
    "session_id": "session-id"
}
```

### 5. Health Check

**GET** `/health`

Check API server health.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2024-01-01T10:00:00Z",
    "version": "0.1.0",
    "orchestrator_status": "healthy"
}
```

## 🔐 Authentication

The API supports optional JWT authentication using Supabase tokens.

### With Authentication

```javascript
// Set auth token
const copilot = new VyuuCopilotClient('http://localhost:8000', 'your-supabase-jwt-token');

// All requests will include the Authorization header
const response = await copilot.sendMessage("Hello");
```

### Without Authentication

```javascript
// No auth token - works for public access
const copilot = new VyuuCopilotClient('http://localhost:8000');

// Requests work without authentication
const response = await copilot.sendMessage("Hello");
```

## 🎯 React Integration Example

Here's a complete React component example:

```jsx
import React, { useState, useEffect } from 'react';

const Chatbot = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [sessionId, setSessionId] = useState(null);

    const sendMessage = async () => {
        if (!input.trim()) return;

        setLoading(true);
        
        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: input,
                    session_id: sessionId,
                    conversation_history: messages,
                }),
            });

            const data = await response.json();
            
            if (data.status === 'success') {
                setMessages(data.conversation_history);
                setSessionId(data.session_id);
                setInput('');
            } else {
                console.error('Chat error:', data.error);
            }
        } catch (error) {
            console.error('Network error:', error);
        } finally {
            setLoading(false);
        }
    };

    const clearChat = async () => {
        if (sessionId) {
            try {
                await fetch(`http://localhost:8000/sessions/${sessionId}`, {
                    method: 'DELETE',
                });
            } catch (error) {
                console.error('Clear session error:', error);
            }
        }
        setMessages([]);
        setSessionId(null);
    };

    return (
        <div className="chatbot">
            <div className="chat-header">
                <h3>Vyuu Copilot</h3>
                <button onClick={clearChat}>Clear Chat</button>
            </div>
            
            <div className="chat-messages">
                {messages.map((msg, index) => (
                    <div key={index} className={`message ${msg.role}`}>
                        <div className="content">{msg.content}</div>
                        <div className="timestamp">
                            {new Date(msg.timestamp).toLocaleTimeString()}
                        </div>
                    </div>
                ))}
            </div>
            
            <div className="chat-input">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder="Ask me about your finances..."
                    disabled={loading}
                />
                <button onClick={sendMessage} disabled={loading || !input.trim()}>
                    {loading ? 'Sending...' : 'Send'}
                </button>
            </div>
        </div>
    );
};

export default Chatbot;
```

## 🐍 Python Integration Example

For Python applications:

```python
import requests
import json
from typing import List, Dict, Optional

class VyuuCopilotClient:
    def __init__(self, base_url: str = "http://localhost:8000", auth_token: Optional[str] = None):
        self.base_url = base_url
        self.auth_token = auth_token
        self.session_id = None
        
    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers
    
    def send_message(self, message: str, conversation_history: List[Dict] = None) -> Dict:
        """Send a message to the copilot."""
        if conversation_history is None:
            conversation_history = []
            
        payload = {
            "message": message,
            "session_id": self.session_id,
            "conversation_history": conversation_history
        }
        
        response = requests.post(
            f"{self.base_url}/chat",
            headers=self._get_headers(),
            json=payload
        )
        
        response.raise_for_status()
        data = response.json()
        
        if data.get("session_id"):
            self.session_id = data["session_id"]
            
        return data
    
    def get_session_history(self) -> List[Dict]:
        """Get conversation history for current session."""
        if not self.session_id:
            return []
            
        response = requests.get(
            f"{self.base_url}/sessions/{self.session_id}/history",
            headers=self._get_headers()
        )
        
        response.raise_for_status()
        return response.json()
    
    def clear_session(self) -> bool:
        """Clear current session."""
        if not self.session_id:
            return True
            
        response = requests.delete(
            f"{self.base_url}/sessions/{self.session_id}",
            headers=self._get_headers()
        )
        
        response.raise_for_status()
        self.session_id = None
        return True

# Usage example
if __name__ == "__main__":
    client = VyuuCopilotClient()
    
    # Send a message
    response = client.send_message("I need help with my budget")
    print(f"Response: {response['response']}")
    
    # Continue conversation
    response = client.send_message("What should I do first?")
    print(f"Response: {response['response']}")
    
    # Get full history
    history = client.get_session_history()
    print(f"Conversation has {len(history)} messages")
```

## 🔧 Configuration

### Environment Variables

Set these environment variables for production:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_ENVIRONMENT=production
API_DEBUG=false

# CORS Configuration (comma-separated)
API_CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/vyuu_copilot

# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key
```

### CORS Configuration

The API automatically configures CORS based on your environment. For development, it allows `localhost:3000` and `localhost:3001`. For production, configure the allowed origins:

```bash
API_CORS_ORIGINS=https://app.vyuu.com,https://vyuu.com
```

## 🚨 Error Handling

The API returns structured error responses:

```json
{
    "error": "Error message",
    "error_code": "ERROR_CODE",
    "details": {
        "additional": "information"
    }
}
```

Common error codes:
- `CHAT_PROCESSING_ERROR`: Error processing the chat request
- `HTTP_ERROR`: General HTTP error
- `INTERNAL_ERROR`: Internal server error
- `TOKEN_VALIDATION_ERROR`: JWT token validation failed

## 📊 Monitoring

The API includes built-in monitoring:

- Request/response logging
- Processing time tracking
- Health check endpoint
- Error tracking and reporting

Access the interactive API documentation at `http://localhost:8000/docs` for testing and exploration.

## 🔄 Session Management

Sessions are automatically managed:

1. **New Session**: If no `session_id` is provided, a new session is created
2. **Session Continuity**: Provide the `session_id` to continue conversations
3. **Session History**: Retrieve full conversation history for any session
4. **Session Cleanup**: Clear sessions when needed

The system maintains conversation context across multiple messages within a session, enabling natural multi-turn conversations.
