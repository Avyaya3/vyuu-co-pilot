# API Architecture Overview

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Chatbot UI    │    │   Mobile App    │    │   Web App       │
│   (React/Vue)   │    │   (React Native)│    │   (Next.js)     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼─────────────┐
                    │     FastAPI Server        │
                    │   (src/api.py)            │
                    │                           │
                    │  • CORS Middleware        │
                    │  • JWT Authentication     │
                    │  • Request Logging        │
                    │  • Error Handling         │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   Main Orchestrator       │
                    │   (src/orchestrator.py)   │
                    │                           │
                    │  • Session Management     │
                    │  • Graph Execution        │
                    │  • Error Recovery         │
                    │  • Performance Tracking   │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   LangGraph System        │
                    │   (src/graphs/)           │
                    │                           │
                    │  • Intent Classification  │
                    │  • Parameter Extraction   │
                    │  • Tool Execution         │
                    │  • Response Synthesis     │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   External Services       │
                    │                           │
                    │  • Supabase Database      │
                    │  • OpenAI API             │
                    │  • Financial Tools        │
                    └───────────────────────────┘
```

## API Endpoints

### Core Chat Endpoints

| Endpoint | Method | Description | Authentication |
|----------|--------|-------------|----------------|
| `/chat` | POST | Main chat endpoint with full conversation support | Optional |
| `/chat/simple` | POST | Simplified chat endpoint for basic integration | Optional |
| `/sessions/{id}/history` | GET | Get conversation history for a session | Optional |
| `/sessions/{id}` | DELETE | Clear conversation history for a session | Optional |

### System Endpoints

| Endpoint | Method | Description | Authentication |
|----------|--------|-------------|----------------|
| `/` | GET | API information and status | None |
| `/health` | GET | Health check and system status | None |
| `/docs` | GET | Interactive API documentation | None |

## Request/Response Flow

### 1. Chat Request Flow

```
User Input → Chatbot UI → FastAPI → Main Orchestrator → LangGraph → Response
     ↓              ↓           ↓              ↓              ↓         ↓
  "Help with    HTTP POST   Validate &    Process &      Execute    Structured
   budget"      /chat       Authenticate   Route          Graph      Response
```

### 2. Session Management Flow

```
New Chat → Generate Session ID → Store State → Process Message → Update History
    ↓              ↓                    ↓              ↓              ↓
  No ID      UUID4()              In-Memory      LangGraph      Return with
  Provided   Generated            Storage        Execution      Updated State
```

### 3. Authentication Flow

```
Request → Check Authorization Header → Verify JWT Token → Extract User Info
    ↓              ↓                        ↓                    ↓
  HTTP         Bearer <token>          Supabase Auth        User Context
  Request      Present?                Validation           Available
```

## Data Models

### ChatRequest
```python
{
    "message": str,                    # User message (1-2000 chars)
    "session_id": Optional[str],       # Session ID for continuity
    "conversation_history": List[ChatMessage],  # Previous messages
    "user_id": Optional[str]           # User ID for personalization
}
```

### ChatResponse
```python
{
    "response": str,                   # Assistant response
    "session_id": str,                 # Session ID
    "status": str,                     # "success" or "error"
    "conversation_history": List[ChatMessage],  # Updated history
    "metadata": Dict[str, Any],        # Additional metadata
    "processing_time_ms": float        # Processing time
}
```

### ChatMessage
```python
{
    "role": str,                       # "user" or "assistant"
    "content": str,                    # Message content
    "timestamp": Optional[str]         # ISO timestamp
}
```

## Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=localhost                    # Server host
API_PORT=8000                        # Server port
API_ENVIRONMENT=development           # Environment
API_DEBUG=true                       # Debug mode
API_CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# Database Configuration
DATABASE_URL=postgresql://...         # PostgreSQL connection
SUPABASE_URL=https://...              # Supabase URL
SUPABASE_KEY=...                      # Supabase anon key
SUPABASE_SERVICE_ROLE_KEY=...         # Supabase service key

# AI Configuration
OPENAI_API_KEY=...                    # OpenAI API key
```

### CORS Configuration

The API automatically configures CORS based on environment:

- **Development**: Allows `localhost:3000`, `localhost:3001`
- **Staging**: Allows `https://staging.vyuu.app`
- **Production**: Allows `https://app.vyuu.com`, `https://vyuu.app`

## Error Handling

### Error Response Format
```python
{
    "error": str,                      # Error message
    "error_code": str,                 # Error code
    "details": Optional[Dict[str, Any]] # Additional details
}
```

### Common Error Codes
- `CHAT_PROCESSING_ERROR`: Error processing chat request
- `TOKEN_VALIDATION_ERROR`: JWT token validation failed
- `SESSION_NOT_FOUND`: Session ID not found
- `HTTP_ERROR`: General HTTP error
- `INTERNAL_ERROR`: Internal server error

## Performance Considerations

### Request Processing
- **Async Processing**: All endpoints use async/await
- **Connection Pooling**: Database connections are pooled
- **Request Logging**: All requests are logged with timing
- **Error Recovery**: Comprehensive error handling and recovery

### Session Management
- **In-Memory Storage**: Sessions stored in memory (Phase 1)
- **Automatic Cleanup**: Sessions cleaned up on completion
- **State Persistence**: Conversation state maintained across requests

### Monitoring
- **Health Checks**: Built-in health check endpoint
- **Processing Time**: Response time tracking
- **Error Tracking**: Comprehensive error logging
- **Request Metrics**: Request/response logging

## Security Features

### Authentication
- **JWT Token Support**: Supabase JWT token validation
- **Optional Authentication**: Endpoints work with or without auth
- **User Context**: User information available when authenticated

### CORS Protection
- **Environment-Based**: CORS configured per environment
- **Origin Validation**: Only allowed origins can access API
- **Credential Support**: Supports authenticated requests

### Input Validation
- **Pydantic Models**: All inputs validated with Pydantic
- **Length Limits**: Message length limits enforced
- **Type Validation**: Strong typing for all inputs

## Deployment

### Development
```bash
python start_api.py
```

### Production
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -e .
CMD ["python", "start_api.py"]
```

## Testing

### Manual Testing
```bash
python test_api.py
```

### API Documentation
- **Interactive Docs**: Available at `/docs`
- **OpenAPI Spec**: Available at `/openapi.json`
- **Health Check**: Available at `/health`

## Integration Examples

### JavaScript/TypeScript
```javascript
const response = await fetch('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: 'Hello' })
});
```

### Python
```python
import requests
response = requests.post('/chat', json={'message': 'Hello'})
```

### cURL
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```
