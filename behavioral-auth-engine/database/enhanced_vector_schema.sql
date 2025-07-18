-- Enhanced Behavioral Vectors Database Schema
-- Supports cumulative, baseline, and session-specific vector storage

-- Drop existing table if it exists (for clean upgrade)
DROP TABLE IF EXISTS enhanced_behavioral_vectors CASCADE;

-- Create enhanced behavioral vectors table
CREATE TABLE enhanced_behavioral_vectors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) NOT NULL,
    session_id TEXT NOT NULL, -- Can be actual session_id or cumulative/baseline identifier
    vector_data FLOAT[] NOT NULL, -- 90-dimensional vector
    vector_type TEXT NOT NULL CHECK (vector_type IN ('session', 'cumulative', 'baseline')),
    confidence_score FLOAT DEFAULT 0.0,
    feature_source TEXT NOT NULL, -- mobile_behavioral_data, cumulative_learning, baseline_creation
    metadata JSONB DEFAULT '{}', -- Flexible metadata storage
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indices for performance
CREATE INDEX idx_enhanced_vectors_user_id ON enhanced_behavioral_vectors(user_id);
CREATE INDEX idx_enhanced_vectors_type ON enhanced_behavioral_vectors(vector_type);
CREATE INDEX idx_enhanced_vectors_user_type ON enhanced_behavioral_vectors(user_id, vector_type);
CREATE INDEX idx_enhanced_vectors_created_at ON enhanced_behavioral_vectors(created_at);

-- Create GIN index for metadata searches
CREATE INDEX idx_enhanced_vectors_metadata ON enhanced_behavioral_vectors USING GIN(metadata);

-- Add vector dimension constraint
ALTER TABLE enhanced_behavioral_vectors 
ADD CONSTRAINT check_vector_dimension 
CHECK (array_length(vector_data, 1) = 90);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for updated_at
CREATE TRIGGER update_enhanced_vectors_updated_at 
    BEFORE UPDATE ON enhanced_behavioral_vectors 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create view for latest cumulative vectors per user
CREATE OR REPLACE VIEW latest_cumulative_vectors AS
SELECT DISTINCT ON (user_id) 
    user_id,
    id,
    vector_data,
    confidence_score,
    metadata,
    created_at
FROM enhanced_behavioral_vectors 
WHERE vector_type = 'cumulative'
ORDER BY user_id, created_at DESC;

-- Create view for baseline vectors per user  
CREATE OR REPLACE VIEW user_baseline_vectors AS
SELECT DISTINCT ON (user_id)
    user_id,
    id,
    vector_data,
    confidence_score,
    metadata,
    created_at
FROM enhanced_behavioral_vectors
WHERE vector_type = 'baseline'
ORDER BY user_id, created_at DESC;

-- Create function to get user vector statistics
CREATE OR REPLACE FUNCTION get_user_vector_stats(target_user_id UUID)
RETURNS TABLE(
    user_id UUID,
    session_vector_count BIGINT,
    cumulative_vector_count BIGINT,
    baseline_vector_count BIGINT,
    latest_cumulative_created TIMESTAMP,
    latest_baseline_created TIMESTAMP,
    total_vectors BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        target_user_id,
        (SELECT COUNT(*) FROM enhanced_behavioral_vectors WHERE user_id = target_user_id AND vector_type = 'session'),
        (SELECT COUNT(*) FROM enhanced_behavioral_vectors WHERE user_id = target_user_id AND vector_type = 'cumulative'),
        (SELECT COUNT(*) FROM enhanced_behavioral_vectors WHERE user_id = target_user_id AND vector_type = 'baseline'),
        (SELECT MAX(created_at) FROM enhanced_behavioral_vectors WHERE user_id = target_user_id AND vector_type = 'cumulative'),
        (SELECT MAX(created_at) FROM enhanced_behavioral_vectors WHERE user_id = target_user_id AND vector_type = 'baseline'),
        (SELECT COUNT(*) FROM enhanced_behavioral_vectors WHERE user_id = target_user_id);
END;
$$ LANGUAGE plpgsql;

-- Create function to clean old session vectors (keep last 100 per user)
CREATE OR REPLACE FUNCTION clean_old_session_vectors()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
    current_deleted INTEGER;
    user_record RECORD;
BEGIN
    -- For each user, keep only the latest 100 session vectors
    FOR user_record IN 
        SELECT DISTINCT user_id FROM enhanced_behavioral_vectors WHERE vector_type = 'session'
    LOOP
        WITH vectors_to_delete AS (
            SELECT id 
            FROM enhanced_behavioral_vectors 
            WHERE user_id = user_record.user_id 
              AND vector_type = 'session'
            ORDER BY created_at DESC
            OFFSET 100
        )
        DELETE FROM enhanced_behavioral_vectors 
        WHERE id IN (SELECT id FROM vectors_to_delete);
        
        GET DIAGNOSTICS current_deleted = ROW_COUNT;
        deleted_count := deleted_count + current_deleted;
    END LOOP;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Update user_profiles table to track vector information
ALTER TABLE user_profiles 
ADD COLUMN IF NOT EXISTS cumulative_vector_count INTEGER DEFAULT 0;

ALTER TABLE user_profiles 
ADD COLUMN IF NOT EXISTS baseline_vector_created TIMESTAMP;

ALTER TABLE user_profiles 
ADD COLUMN IF NOT EXISTS last_vector_update TIMESTAMP DEFAULT NOW();

-- Create trigger to update user_profiles when vectors are added
CREATE OR REPLACE FUNCTION update_user_profile_vector_stats()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.vector_type = 'cumulative' THEN
        UPDATE user_profiles 
        SET 
            cumulative_vector_count = COALESCE(cumulative_vector_count, 0) + 1,
            last_vector_update = NOW()
        WHERE user_id = NEW.user_id;
    ELSIF NEW.vector_type = 'baseline' THEN
        UPDATE user_profiles 
        SET 
            baseline_vector_created = NOW(),
            last_vector_update = NOW()
        WHERE user_id = NEW.user_id;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_user_profiles_on_vector_insert
    AFTER INSERT ON enhanced_behavioral_vectors
    FOR EACH ROW
    EXECUTE FUNCTION update_user_profile_vector_stats();

-- Migrate existing behavioral_vectors data (if needed)
INSERT INTO enhanced_behavioral_vectors (
    user_id, 
    session_id, 
    vector_data, 
    vector_type, 
    confidence_score, 
    feature_source,
    metadata,
    created_at
)
SELECT 
    user_id,
    session_id::TEXT,
    vector_data,
    'session',
    confidence_score,
    COALESCE(feature_source, 'legacy_migration'),
    jsonb_build_object('migrated_from_legacy', true),
    created_at
FROM behavioral_vectors
WHERE NOT EXISTS (
    SELECT 1 FROM enhanced_behavioral_vectors 
    WHERE enhanced_behavioral_vectors.user_id = behavioral_vectors.user_id 
    AND enhanced_behavioral_vectors.session_id = behavioral_vectors.session_id::TEXT
);

-- Create summary view for monitoring
CREATE OR REPLACE VIEW vector_storage_summary AS
SELECT 
    vector_type,
    COUNT(*) as total_vectors,
    COUNT(DISTINCT user_id) as unique_users,
    AVG(confidence_score) as avg_confidence,
    MIN(created_at) as earliest_vector,
    MAX(created_at) as latest_vector
FROM enhanced_behavioral_vectors
GROUP BY vector_type;

-- Create performance monitoring view
CREATE OR REPLACE VIEW vector_performance_metrics AS
SELECT 
    DATE_TRUNC('day', created_at) as date,
    vector_type,
    COUNT(*) as vectors_created,
    COUNT(DISTINCT user_id) as active_users,
    AVG(confidence_score) as avg_confidence
FROM enhanced_behavioral_vectors
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY DATE_TRUNC('day', created_at), vector_type
ORDER BY date DESC, vector_type;

-- Grant appropriate permissions
GRANT SELECT, INSERT, UPDATE ON enhanced_behavioral_vectors TO authenticated;
GRANT SELECT ON latest_cumulative_vectors TO authenticated;
GRANT SELECT ON user_baseline_vectors TO authenticated;
GRANT SELECT ON vector_storage_summary TO authenticated;
GRANT SELECT ON vector_performance_metrics TO authenticated;
GRANT EXECUTE ON FUNCTION get_user_vector_stats(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION clean_old_session_vectors() TO authenticated;
