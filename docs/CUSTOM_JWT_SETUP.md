# Custom JWT Authentication Setup

## Overview

The authentication system has been simplified to support **only custom JWT tokens from NextJS**. This provides maximum performance and eliminates unnecessary Supabase JWT verification attempts.

## Required Environment Variables

```bash
# Required: Custom JWT secret for NextJS tokens
CUSTOM_JWT_SECRET=your-secure-jwt-secret-key-here

# Optional: JWT validation settings
CUSTOM_JWT_ISSUER=https://your-nextjs-app.com
CUSTOM_JWT_AUDIENCE=vyuu-copilot-api
CUSTOM_JWT_ALGORITHM=HS256
CUSTOM_JWT_EXPIRATION_HOURS=24

# Required: Supabase configuration (for MCP calls)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key

# Required: Other services
OPENAI_API_KEY=your-openai-api-key
DATABASE_URL=postgresql://user:password@localhost:5432/database
```

## NextJS JWT Token Creation

### 1. Install JWT Library
```bash
npm install jsonwebtoken
npm install @types/jsonwebtoken  # For TypeScript
```

### 2. Create JWT Token
```javascript
// NextJS side - Create custom JWT token
import jwt from 'jsonwebtoken';

const customJWTSecret = process.env.CUSTOM_JWT_SECRET;
const userData = {
  sub: userId,                    // Required: User ID
  email: userEmail,              // Optional: User email
  role: 'authenticated',         // Optional: User role
  iss: 'https://your-nextjs-app.com',  // Optional: Issuer
  aud: 'vyuu-copilot-api',       // Optional: Audience
  iat: Math.floor(Date.now() / 1000),
  exp: Math.floor(Date.now() / 1000) + (24 * 60 * 60) // 24 hours
};

const token = jwt.sign(userData, customJWTSecret, { algorithm: 'HS256' });
```

### 3. Send Request with JWT
```javascript
// NextJS side - Send request with custom JWT
const response = await fetch('/api/chat', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    message: 'Show me my financial assets',
    session_id: 'session-123'
  })
});
```

## FastAPI Integration

The FastAPI server will:

1. ✅ **Extract JWT token** from `Authorization: Bearer <token>` header
2. ✅ **Verify JWT signature** using `CUSTOM_JWT_SECRET`
3. ✅ **Extract user_id** from the `sub` claim
4. ✅ **Mint Supabase JWT** for MCP tool calls
5. ✅ **Process request** through LangGraph with user context

## Expected JWT Payload

```json
{
  "sub": "user-id-123",           // Required: User ID
  "email": "user@example.com",    // Optional: User email
  "role": "authenticated",        // Optional: User role
  "iss": "https://your-nextjs-app.com",  // Optional: Issuer
  "aud": "vyuu-copilot-api",      // Optional: Audience
  "iat": 1234567890,              // Required: Issued at
  "exp": 1234654290               // Required: Expiration
}
```

## Authentication Flow

```
NextJS App → Custom JWT → FastAPI → Verify JWT → Extract user_id → 
Mint Supabase JWT → LangGraph → MCP Tools → Supabase RLS
```

## Testing

Run the simplified test to verify your setup:

```bash
python test_simplified_custom_jwt_flow.py
```

## Security Notes

- ✅ **JWT secret is never exposed** to the frontend
- ✅ **Supabase JWT secret is never exposed** to the frontend
- ✅ **Row-level security** is enforced via Supabase JWT for MCP calls
- ✅ **Token expiration** is enforced
- ✅ **Signature verification** prevents token tampering

## Migration from Supabase JWT

If you were previously using Supabase JWT tokens directly:

1. **Remove Supabase JWT logic** from your frontend
2. **Implement custom JWT creation** in your NextJS backend
3. **Set CUSTOM_JWT_SECRET** environment variable
4. **Update frontend** to send custom JWT tokens
5. **Test the flow** with the provided test suite

## Troubleshooting

### Common Issues

1. **"Custom JWT configuration not available"**
   - Ensure `CUSTOM_JWT_SECRET` is set in environment variables
   - Restart the FastAPI server after setting environment variables

2. **"Invalid token"**
   - Check that JWT secret matches between NextJS and FastAPI
   - Verify token hasn't expired
   - Ensure token is properly formatted

3. **"Token missing user ID"**
   - Ensure JWT payload includes `sub` claim with user ID
   - Check that `sub` claim is not empty

### Debug Mode

Enable debug logging to see JWT verification details:

```bash
export LOG_LEVEL=DEBUG
```

## Performance Benefits

- ✅ **No wasted Supabase JWT verification attempts**
- ✅ **Single authentication path** (custom JWT only)
- ✅ **Faster token verification**
- ✅ **Simplified error handling**
- ✅ **Clean, maintainable code**

## Ready for Production

The simplified authentication system is now ready for production use with NextJS integration. Just set the required environment variables and implement JWT token creation in your NextJS backend.
