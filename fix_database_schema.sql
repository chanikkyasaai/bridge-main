-- Add missing vectors_collected column to user_profiles table
ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS vectors_collected INTEGER DEFAULT 0;

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_user_profiles_vectors_collected ON user_profiles(vectors_collected);

-- Add constraint to ensure authentication_decisions.decision matches allowed values
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM information_schema.table_constraints 
        WHERE constraint_name = 'authentication_decisions_decision_check'
        AND table_name = 'authentication_decisions'
    ) THEN
        ALTER TABLE authentication_decisions 
        ADD CONSTRAINT authentication_decisions_decision_check 
        CHECK (decision IN ('allow', 'challenge', 'block', 'learn'));
    END IF;
END $$;

-- Update any existing null vectors_collected to 0
UPDATE user_profiles SET vectors_collected = 0 WHERE vectors_collected IS NULL;

-- Verify the changes
SELECT column_name, data_type, is_nullable, column_default 
FROM information_schema.columns 
WHERE table_name = 'user_profiles' 
AND column_name = 'vectors_collected';
