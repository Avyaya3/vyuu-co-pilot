# Supabase Connection Setup Guide

This guide walks you through setting up and testing your Supabase connection for the Vyuu Copilot v2 project.

## 📋 Prerequisites

1. **Existing Supabase Project**: You should already have a Supabase project created
2. **Python 3.9+**: Ensure you have Python 3.9 or higher installed
3. **Project Dependencies**: Install the project dependencies

## 🔧 Setup Instructions

### Step 1: Install Dependencies

```bash
# Install the project and its dependencies
pip install -e .

# Or if you're developing
pip install -e ".[dev]"
```

### Step 2: Configure Environment Variables

1. **Copy the environment template:**
   ```bash
   cp .env.template .env
   ```

2. **Update your `.env` file with your Supabase credentials:**

   ```env
   # Supabase Configuration
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_KEY=your_anon_public_key_here
   SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here

   # Database Configuration (PostgreSQL via Supabase)
   DATABASE_URL=postgresql://postgres:your_password@db.your-project-id.supabase.co:5432/postgres
   DB_POOL_SIZE=10
   DB_MAX_OVERFLOW=20
   DB_POOL_TIMEOUT=30
   DB_POOL_RECYCLE=3600

   # Supabase Connection Settings
   SUPABASE_TIMEOUT=30
   SUPABASE_MAX_RETRIES=3
   SUPABASE_RETRY_DELAY=1

   # Authentication
   JWT_SECRET_KEY=your_32_character_secret_key_here_minimum_length
   JWT_ALGORITHM=HS256
   JWT_EXPIRATION_HOURS=24
   ```

### Step 3: Find Your Supabase Credentials

#### From Supabase Dashboard:

1. **Go to your Supabase project dashboard**
2. **Navigate to Settings → API**
3. **Copy the required values:**
   - **Project URL** → `SUPABASE_URL`
   - **anon public key** → `SUPABASE_KEY`
   - **service_role secret key** → `SUPABASE_SERVICE_ROLE_KEY`

#### Database URL:
1. **Go to Settings → Database**
2. **Find the connection string under "Connection string"**
3. **Use the format:** `postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT-REF].supabase.co:5432/postgres`

### Step 4: Generate JWT Secret Key

Generate a secure JWT secret key (minimum 32 characters):

```bash
# Option 1: Using Python
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Option 2: Using openssl
openssl rand -base64 32

# Option 3: Using uuidgen (macOS/Linux)
uuidgen | tr -d '-' | head -c 32 && echo
```

## ✅ Testing Your Connection

### Quick Test

Run the comprehensive connection test script:

```bash
python scripts/test_connection.py
```

This will test:
- ✅ Configuration loading
- ✅ Supabase client initialization
- ✅ Database connectivity
- ✅ Connection pooling
- ✅ JWT token verification
- ✅ Health checks

### Expected Output

```
🔧 Vyuu Copilot v2 - Supabase Connection Test
==================================================
🚀 Starting Supabase connection tests...
==================================================

🔍 Running Configuration test...
✅ Configuration Loading: PASSED

🔍 Running Supabase Client test...
✅ Supabase Client: PASSED

🔍 Running Database Connection test...
✅ Database Connection: PASSED

🔍 Running Connection Pool test...
✅ Connection Pool: PASSED

🔍 Running JWT Verification test...
✅ JWT Token Verification: PASSED

🔍 Running Health Check test...
✅ Health Check: PASSED

==================================================
📊 TEST SUMMARY
==================================================
Total tests: 6
Passed: 6 ✅
Failed: 0 ❌
Success rate: 100.0%

🎉 All tests passed! Your Supabase connection is working correctly.
```

## 🔍 Manual Testing

### Test Configuration Loading

```python
from src.utils.config import get_config

config = get_config()
print(f"Supabase URL: {config.supabase.url}")
print(f"Pool Size: {config.database.pool_size}")
```

### Test Database Connection

```python
import asyncio
from src.utils.database import get_db_client

async def test_db():
    db = get_db_client()
    result = await db.execute_query("SELECT NOW() as current_time", fetch_one=True)
    print(f"Database time: {result['current_time']}")

asyncio.run(test_db())
```

### Test Authentication

```python
from src.utils.auth import get_auth_manager

auth = get_auth_manager()
# Test token extraction from Authorization header
test_token = auth.extract_token_from_header("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
print(f"Extracted token: {test_token}")
```

## 🚨 Troubleshooting

### Common Issues

#### 1. **Configuration Not Found**
```
Error: Failed to import modules
```
**Solution:** Make sure you're running from the project root directory

#### 2. **Invalid Supabase URL**
```
Error: Invalid Supabase URL format
```
**Solution:** Ensure your URL follows the format: `https://your-project-id.supabase.co`

#### 3. **Database Connection Failed**
```
Error: Pool creation failed
```
**Solutions:**
- Check your `DATABASE_URL` format
- Verify your database password
- Ensure your Supabase project is active
- Check firewall/network settings

#### 4. **Authentication Errors**
```
Error: JWT secret key must be at least 32 characters long
```
**Solution:** Generate a proper JWT secret key (see Step 4 above)

#### 5. **Import Errors**
```
ModuleNotFoundError: No module named 'supabase'
```
**Solution:** Install dependencies:
```bash
pip install -e .
```

### Debug Mode

For detailed debugging, set environment variables:

```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
python scripts/test_connection.py
```

### Check Dependencies

Verify all required packages are installed:

```bash
pip list | grep -E "(supabase|asyncpg|tenacity|pyjwt|gotrue)"
```

## 🔒 Security Best Practices

1. **Never commit `.env` file** - It's already in `.gitignore`
2. **Use environment-specific configurations** for different deployments
3. **Rotate JWT secret keys** regularly in production
4. **Use service role key sparingly** - prefer anon key when possible
5. **Enable Row Level Security (RLS)** in your Supabase tables
6. **Monitor connection usage** and adjust pool sizes as needed

## 📊 Connection Monitoring

The system includes built-in health checks and connection monitoring:

```python
import asyncio
from src.utils.database import get_db_client

async def check_health():
    db = get_db_client()
    health = await db.health_check()
    print(f"Status: {health['status']}")
    print(f"Connection Stats: {health['connection_stats']}")

asyncio.run(check_health())
```

## 🎯 Next Steps

Once your connection is working:

1. **Test with your actual data** - Try querying your existing tables
2. **Set up authentication flows** - Test user registration/login
3. **Configure production settings** - Adjust pool sizes and timeouts
4. **Implement error handling** - Add application-specific error handling
5. **Monitor performance** - Set up logging and monitoring

## 📞 Support

If you encounter issues:

1. Check the [test results JSON file](test_results.json) for detailed error information
2. Review Supabase logs in your dashboard
3. Check the [Supabase documentation](https://supabase.com/docs)
4. Verify your project settings and credentials

---

**Congratulations!** 🎉 Your Supabase connection is now ready for the LangGraph intent orchestration system. 