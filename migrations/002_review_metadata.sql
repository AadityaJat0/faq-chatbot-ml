-- ResolveBot migration 002: owner review metadata
--
-- Safe to run more than once, and safe to run against a deployment that is
-- already serving traffic. Every statement is guarded so an existing
-- knowledge_suggestions table keeps working unchanged.
--
-- Run it in: Supabase dashboard -> SQL Editor -> New query -> Run.

-- 1. Columns the owner review page writes when a decision is recorded.
alter table public.knowledge_suggestions
    add column if not exists reviewed_at timestamptz;

alter table public.knowledge_suggestions
    add column if not exists reviewer_note text;

-- 2. Every new suggestion starts life awaiting review.
alter table public.knowledge_suggestions
    alter column status set default 'pending';

update public.knowledge_suggestions
    set status = 'pending'
    where status is null;

-- 3. Restrict status to the three review states. Added only if absent so the
--    migration stays idempotent.
do $$
begin
    if not exists (
        select 1 from pg_constraint
        where conname = 'knowledge_suggestions_status_check'
    ) then
        alter table public.knowledge_suggestions
            add constraint knowledge_suggestions_status_check
            check (status in ('pending', 'approved', 'rejected'));
    end if;
end
$$;

-- 4. The review page always queries "pending, newest first".
create index if not exists knowledge_suggestions_status_created_idx
    on public.knowledge_suggestions (status, created_at desc);

-- 5. Keep public clients out of this table. The application reaches it with
--    the service-role key held in Streamlit Secrets, which bypasses RLS
--    server-side; no anon-key policy is created here on purpose.
alter table public.knowledge_suggestions enable row level security;
