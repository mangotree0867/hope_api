-- User Management Schema for PostgreSQL
-- Includes user table with secure password handling

-- Drop existing table if needed (comment out in production)
DROP TABLE IF EXISTS users CASCADE;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,                          -- Auto-incrementing primary key
    name VARCHAR(255) NOT NULL,                     -- User's display name
    login_id VARCHAR(100) UNIQUE NOT NULL,          -- Unique login identifier (username/email)
    password_hash VARCHAR(255) NOT NULL,            -- Hashed password (SHA256)
    email VARCHAR(255) NOT NULL,                    -- User Email
    salt VARCHAR(32) NOT NULL,                      -- Salt for password hashing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_login_id ON users(login_id);
CREATE INDEX IF NOT EXISTS idx_users_name ON users(name);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- Optional: Create a user sessions table for tracking active sessions
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    is_valid BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);