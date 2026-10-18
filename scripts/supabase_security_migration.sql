-- ==============================================================================
-- TalentAI-X: Security & RBAC Migration for Supabase PostgreSQL
-- Run this script in your Supabase SQL Editor (Dashboard -> SQL Editor -> New Query)
-- ==============================================================================

-- 1. Create user_profiles table in public schema
CREATE TABLE IF NOT EXISTS public.user_profiles (
    id UUID PRIMARY KEY,
    email VARCHAR(255),
    role VARCHAR(20) NOT NULL DEFAULT 'individual',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_profiles_email ON public.user_profiles(email);
CREATE INDEX IF NOT EXISTS idx_user_profiles_role ON public.user_profiles(role);

-- 2. Drop FK to auth.users if it exists — system/API-key accounts use
--    synthetic UUIDs that are not in auth.users. Supabase JWT validation
--    at the application layer ensures real user identity. The trigger in
--    step 4 still syncs sign-ups from auth.users into user_profiles.
ALTER TABLE IF EXISTS public.user_profiles
    DROP CONSTRAINT IF EXISTS fk_user_profiles_auth_users;

-- 3. Populate existing users from auth.users (if any exist already)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables 
        WHERE table_schema = 'auth' AND table_name = 'users'
    ) THEN
        INSERT INTO public.user_profiles (id, email, role, created_at, updated_at)
        SELECT 
            id, 
            email, 
            'individual', 
            COALESCE(created_at, NOW()), 
            NOW()
        FROM auth.users
        ON CONFLICT (id) DO UPDATE 
        SET email = EXCLUDED.email, updated_at = NOW();
    END IF;
END $$;

-- 4. Automatic Supabase Trigger: create user_profile whenever a user signs up
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_profiles (id, email, role, created_at, updated_at)
    VALUES (NEW.id, NEW.email, 'individual', NOW(), NOW())
    ON CONFLICT (id) DO UPDATE 
    SET email = EXCLUDED.email, updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Bind trigger to auth.users
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- 5. Add user_id ownership fields to application tables
ALTER TABLE IF EXISTS public.parse_jobs 
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES public.user_profiles(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_parse_jobs_user_id ON public.parse_jobs(user_id);

ALTER TABLE IF EXISTS public.candidates 
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES public.user_profiles(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_candidates_user_id ON public.candidates(user_id);

ALTER TABLE IF EXISTS public.jobs 
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES public.user_profiles(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON public.jobs(user_id);

ALTER TABLE IF EXISTS public.match_results 
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES public.user_profiles(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_match_results_user_id ON public.match_results(user_id);

ALTER TABLE IF EXISTS public.api_keys 
    ADD COLUMN IF NOT EXISTS user_id UUID REFERENCES public.user_profiles(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON public.api_keys(user_id);
