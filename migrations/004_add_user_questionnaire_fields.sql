-- Migration: Add questionnaire fields to users table
-- This migration adds fields to store user preferences and questionnaire data

-- Add questionnaire fields to users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS industry VARCHAR(100);
ALTER TABLE users ADD COLUMN IF NOT EXISTS job_title VARCHAR(100);
ALTER TABLE users ADD COLUMN IF NOT EXISTS experience_level VARCHAR(50);
ALTER TABLE users ADD COLUMN IF NOT EXISTS content_goals JSONB DEFAULT '[]'::jsonb;
ALTER TABLE users ADD COLUMN IF NOT EXISTS target_audience VARCHAR(100);
ALTER TABLE users ADD COLUMN IF NOT EXISTS preferred_tone VARCHAR(50);
ALTER TABLE users ADD COLUMN IF NOT EXISTS content_frequency VARCHAR(50);
ALTER TABLE users ADD COLUMN IF NOT EXISTS topics_of_interest JSONB DEFAULT '[]'::jsonb;
ALTER TABLE users ADD COLUMN IF NOT EXISTS writing_style_preference VARCHAR(50);
ALTER TABLE users ADD COLUMN IF NOT EXISTS blog_length_preference VARCHAR(50);
ALTER TABLE users ADD COLUMN IF NOT EXISTS include_images BOOLEAN DEFAULT true;
ALTER TABLE users ADD COLUMN IF NOT EXISTS include_data_visualizations BOOLEAN DEFAULT false;
ALTER TABLE users ADD COLUMN IF NOT EXISTS seo_focus BOOLEAN DEFAULT true;
ALTER TABLE users ADD COLUMN IF NOT EXISTS questionnaire_completed BOOLEAN DEFAULT false;
ALTER TABLE users ADD COLUMN IF NOT EXISTS questionnaire_completed_at TIMESTAMP WITH TIME ZONE;

-- Create indexes for commonly queried fields
CREATE INDEX IF NOT EXISTS idx_users_industry ON users(industry);
CREATE INDEX IF NOT EXISTS idx_users_experience_level ON users(experience_level);
CREATE INDEX IF NOT EXISTS idx_users_questionnaire_completed ON users(questionnaire_completed);

-- Add comments for documentation
COMMENT ON COLUMN users.industry IS 'User industry/sector (e.g., Technology, Healthcare, Finance)';
COMMENT ON COLUMN users.job_title IS 'User job title or role';
COMMENT ON COLUMN users.experience_level IS 'Experience level (Beginner, Intermediate, Advanced, Expert)';
COMMENT ON COLUMN users.content_goals IS 'JSON array of content goals (e.g., ["Lead Generation", "Brand Awareness"])';
COMMENT ON COLUMN users.target_audience IS 'Primary target audience for content';
COMMENT ON COLUMN users.preferred_tone IS 'Preferred writing tone (Professional, Casual, Technical, etc.)';
COMMENT ON COLUMN users.content_frequency IS 'How often user plans to create content';
COMMENT ON COLUMN users.topics_of_interest IS 'JSON array of topics user is interested in';
COMMENT ON COLUMN users.writing_style_preference IS 'Preferred writing style';
COMMENT ON COLUMN users.blog_length_preference IS 'Preferred blog post length (Short, Medium, Long)';
COMMENT ON COLUMN users.include_images IS 'Whether to include images in generated content';
COMMENT ON COLUMN users.include_data_visualizations IS 'Whether to include charts/graphs';
COMMENT ON COLUMN users.seo_focus IS 'Whether to optimize content for SEO';
COMMENT ON COLUMN users.questionnaire_completed IS 'Whether user has completed the onboarding questionnaire';
COMMENT ON COLUMN users.questionnaire_completed_at IS 'When the questionnaire was completed'; 