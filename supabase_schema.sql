-- ResolveBot persistent-chat schema
-- Run this once in the Supabase SQL Editor before adding Streamlit Secrets.

create extension if not exists pgcrypto;

create table if not exists public.conversations (
  id uuid primary key default gen_random_uuid(),
  visitor_token_hash text not null unique,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.messages (
  id uuid primary key,
  conversation_id uuid not null references public.conversations(id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null check (char_length(content) <= 4000),
  created_at timestamptz not null default now()
);

create index if not exists messages_conversation_created_at_idx
  on public.messages (conversation_id, created_at);

-- Suggestions are deliberately separate from the trusted FAQ training data.
-- They are pending until the project owner reviews them.
create table if not exists public.knowledge_suggestions (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid references public.conversations(id) on delete set null,
  question text not null check (char_length(question) <= 4000),
  suggested_answer text not null check (char_length(suggested_answer) <= 4000),
  status text not null default 'pending' check (status in ('pending', 'approved', 'rejected')),
  created_at timestamptz not null default now()
);

create index if not exists knowledge_suggestions_status_created_at_idx
  on public.knowledge_suggestions (status, created_at);

-- The Streamlit server uses the service-role key from its private Secrets.
-- Keep the public API closed: browser users must never have direct table access.
alter table public.conversations enable row level security;
alter table public.messages enable row level security;
alter table public.knowledge_suggestions enable row level security;

revoke all on table public.conversations from anon, authenticated;
revoke all on table public.messages from anon, authenticated;
revoke all on table public.knowledge_suggestions from anon, authenticated;

grant all on table public.conversations to service_role;
grant all on table public.messages to service_role;
grant all on table public.knowledge_suggestions to service_role;
