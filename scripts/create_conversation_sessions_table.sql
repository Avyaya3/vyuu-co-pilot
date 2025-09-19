-- Create conversation_sessions table for LangGraph state persistence
-- This table stores conversation session data to enable persistence across server restarts

CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255),
    state_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    message_count INTEGER DEFAULT 0,
    last_intent VARCHAR(100),
    last_confidence DECIMAL(3,2)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_conversation_sessions_user_id ON conversation_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_conversation_sessions_created_at ON conversation_sessions(created_at);
CREATE INDEX IF NOT EXISTS idx_conversation_sessions_updated_at ON conversation_sessions(updated_at);
CREATE INDEX IF NOT EXISTS idx_conversation_sessions_expires_at ON conversation_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_conversation_sessions_is_active ON conversation_sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_conversation_sessions_user_active ON conversation_sessions(user_id, is_active);

-- Create a function to automatically update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_conversation_sessions_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
DROP TRIGGER IF EXISTS trigger_update_conversation_sessions_updated_at ON conversation_sessions;
CREATE TRIGGER trigger_update_conversation_sessions_updated_at
    BEFORE UPDATE ON conversation_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_conversation_sessions_updated_at();

-- Add comments for documentation
COMMENT ON TABLE conversation_sessions IS 'Stores conversation session data for LangGraph state persistence';
COMMENT ON COLUMN conversation_sessions.session_id IS 'Unique session identifier';
COMMENT ON COLUMN conversation_sessions.user_id IS 'Associated user ID (nullable for anonymous sessions)';
COMMENT ON COLUMN conversation_sessions.state_data IS 'Serialized LangGraph state data as JSON';
COMMENT ON COLUMN conversation_sessions.created_at IS 'Session creation timestamp';
COMMENT ON COLUMN conversation_sessions.updated_at IS 'Last update timestamp (auto-updated)';
COMMENT ON COLUMN conversation_sessions.expires_at IS 'Session expiration timestamp (nullable)';
COMMENT ON COLUMN conversation_sessions.is_active IS 'Whether the session is active';
COMMENT ON COLUMN conversation_sessions.message_count IS 'Number of messages in the session';
COMMENT ON COLUMN conversation_sessions.last_intent IS 'Last classified intent';
COMMENT ON COLUMN conversation_sessions.last_confidence IS 'Last intent confidence score';
