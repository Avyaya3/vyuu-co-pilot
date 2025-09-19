# Authentication Integration Examples

## Option 1: Use Existing Supabase JWT Tokens (Recommended)

If your chatbot app already uses Supabase authentication, you can simply pass the existing JWT tokens to the API.

### Frontend Integration (React/Next.js)

```javascript
// In your chatbot app, you already have Supabase auth
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY)

// Your existing auth setup
const { data: { session } } = await supabase.auth.getSession()

// Use the same token for the copilot API
const copilotResponse = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${session.access_token}` // Use your existing token
    },
    body: JSON.stringify({
        message: "I need help with my budget",
        user_id: session.user.id // Optional: pass user ID for personalization
    })
})
```

### Vue.js Integration

```javascript
// In your Vue chatbot app
import { supabase } from '@/lib/supabase'

export default {
    async sendMessage(message) {
        const { data: { session } } = await supabase.auth.getSession()
        
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${session.access_token}`
            },
            body: JSON.stringify({
                message,
                user_id: session.user.id
            })
        })
        
        return await response.json()
    }
}
```

### Angular Integration

```typescript
// In your Angular chatbot app
import { SupabaseService } from './services/supabase.service'

@Injectable()
export class CopilotService {
    constructor(private supabase: SupabaseService) {}
    
    async sendMessage(message: string) {
        const session = await this.supabase.getSession()
        
        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${session.access_token}`
            },
            body: JSON.stringify({
                message,
                user_id: session.user.id
            })
        })
        
        return await response.json()
    }
}
```

## Option 2: Custom Authentication Headers

If you want to use a different authentication method, you can modify the API to accept custom headers.

### Custom Header Authentication

```javascript
// Send custom user information
const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-User-ID': 'your-user-id',
        'X-User-Email': 'user@example.com',
        'X-User-Role': 'premium'
    },
    body: JSON.stringify({
        message: "I need help with my budget"
    })
})
```

## Option 3: API Key Authentication

For server-to-server communication or simple authentication.

### API Key Setup

```javascript
// Use API key instead of JWT
const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'your-api-key',
        'X-User-ID': 'user-id'
    },
    body: JSON.stringify({
        message: "I need help with my budget"
    })
})
```

## Option 4: No Authentication (Public Access)

For public chatbots or development/testing.

```javascript
// No authentication required
const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        message: "I need help with my budget"
    })
})
```

## Complete Integration Example

Here's a complete example showing how to integrate with your existing Supabase auth:

```javascript
class CopilotIntegration {
    constructor(supabaseClient, apiBaseUrl = 'http://localhost:8000') {
        this.supabase = supabaseClient
        this.apiBaseUrl = apiBaseUrl
        this.sessionId = null
    }
    
    async sendMessage(message, conversationHistory = []) {
        try {
            // Get current session from your existing Supabase auth
            const { data: { session }, error } = await this.supabase.auth.getSession()
            
            if (error) {
                console.error('Auth error:', error)
                // Fallback to unauthenticated request
                return await this.sendUnauthenticatedMessage(message, conversationHistory)
            }
            
            const headers = {
                'Content-Type': 'application/json'
            }
            
            // Add authentication if session exists
            if (session?.access_token) {
                headers['Authorization'] = `Bearer ${session.access_token}`
            }
            
            const response = await fetch(`${this.apiBaseUrl}/chat`, {
                method: 'POST',
                headers,
                body: JSON.stringify({
                    message,
                    session_id: this.sessionId,
                    conversation_history: conversationHistory,
                    user_id: session?.user?.id // Optional: for personalization
                })
            })
            
            if (!response.ok) {
                throw new Error(`API Error: ${response.status}`)
            }
            
            const data = await response.json()
            
            // Store session ID for conversation continuity
            if (data.session_id) {
                this.sessionId = data.session_id
            }
            
            return data
            
        } catch (error) {
            console.error('Copilot API error:', error)
            throw error
        }
    }
    
    async sendUnauthenticatedMessage(message, conversationHistory = []) {
        // Fallback for when authentication fails
        const response = await fetch(`${this.apiBaseUrl}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message,
                session_id: this.sessionId,
                conversation_history: conversationHistory
            })
        })
        
        const data = await response.json()
        
        if (data.session_id) {
            this.sessionId = data.session_id
        }
        
        return data
    }
    
    async getSessionHistory() {
        if (!this.sessionId) return []
        
        const { data: { session } } = await this.supabase.auth.getSession()
        
        const headers = {}
        if (session?.access_token) {
            headers['Authorization'] = `Bearer ${session.access_token}`
        }
        
        const response = await fetch(`${this.apiBaseUrl}/sessions/${this.sessionId}/history`, {
            headers
        })
        
        return await response.json()
    }
}

// Usage in your chatbot app
const copilot = new CopilotIntegration(supabase)

// Send a message (automatically uses your existing auth)
const response = await copilot.sendMessage("I need help with my budget")
console.log(response.response)
```

## React Hook Example

```javascript
import { useState, useEffect } from 'react'
import { supabase } from './lib/supabase'

export const useCopilot = () => {
    const [sessionId, setSessionId] = useState(null)
    const [loading, setLoading] = useState(false)
    
    const sendMessage = async (message, conversationHistory = []) => {
        setLoading(true)
        
        try {
            const { data: { session } } = await supabase.auth.getSession()
            
            const headers = {
                'Content-Type': 'application/json'
            }
            
            if (session?.access_token) {
                headers['Authorization'] = `Bearer ${session.access_token}`
            }
            
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers,
                body: JSON.stringify({
                    message,
                    session_id: sessionId,
                    conversation_history,
                    user_id: session?.user?.id
                })
            })
            
            const data = await response.json()
            
            if (data.session_id) {
                setSessionId(data.session_id)
            }
            
            return data
            
        } catch (error) {
            console.error('Copilot error:', error)
            throw error
        } finally {
            setLoading(false)
        }
    }
    
    return { sendMessage, loading, sessionId }
}

// Usage in your component
const Chatbot = () => {
    const { sendMessage, loading } = useCopilot()
    const [messages, setMessages] = useState([])
    
    const handleSendMessage = async (message) => {
        const response = await sendMessage(message, messages)
        setMessages(response.conversation_history)
    }
    
    return (
        <div>
            {/* Your chatbot UI */}
        </div>
    )
}
```

## Environment Configuration

Make sure your API server has the same Supabase configuration as your chatbot app:

```bash
# .env file for the API server
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
DATABASE_URL=postgresql://user:password@localhost:5432/vyuu_copilot
```

## Benefits of This Approach

1. **Seamless Integration**: Uses your existing auth system
2. **User Context**: The API gets user information for personalization
3. **Security**: Leverages your existing security setup
4. **Consistency**: Same auth across your entire application
5. **Fallback Support**: Works even if auth fails (for public access)

The API is designed to work with or without authentication, so you can start with Option 1 (using your existing Supabase tokens) and it will work immediately!


