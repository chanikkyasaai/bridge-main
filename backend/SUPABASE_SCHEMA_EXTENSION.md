# Supabase Database Schema Extension for Behavioral Authentication

## Additional Tables Required

### 1. User Profiles (for ML Engine)
```sql
create table user_profiles (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid references users(id) not null,
  current_session_count integer default 0,
  total_sessions integer default 0,
  current_phase text default 'learning', -- learning, gradual_risk, full_auth
  risk_score float default 0.0,
  last_activity timestamp default now(),
  behavioral_model_version integer default 1,
  created_at timestamp default now(),
  updated_at timestamp default now()
);
```

### 2. Behavioral Vectors
```sql
create table behavioral_vectors (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid references users(id) not null,
  session_id uuid references sessions(id) not null,
  vector_data float[] not null, -- 90-dimensional vector
  confidence_score float default 0.0,
  feature_source text not null, -- typing, touch, navigation, contextual
  created_at timestamp default now()
);

-- Create index for fast vector retrieval
create index idx_behavioral_vectors_user_id on behavioral_vectors(user_id);
create index idx_behavioral_vectors_session_id on behavioral_vectors(session_id);
```

### 3. Authentication Decisions (ML Layer Results)
```sql
create table authentication_decisions (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid references users(id) not null,
  session_id uuid references sessions(id) not null,
  decision text not null, -- allow, challenge, block, learn
  confidence float not null,
  similarity_score float,
  layer_used text not null, -- faiss, adaptive
  risk_factors text[], -- array of risk factors
  threshold_used float,
  processing_time_ms integer,
  created_at timestamp default now()
);
```

### 4. Behavioral Feedback
```sql
create table behavioral_feedback (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid references users(id) not null,
  session_id uuid references sessions(id) not null,
  decision_id uuid references authentication_decisions(id) not null,
  was_correct boolean not null,
  feedback_source text not null, -- user_report, timeout, escalation
  corrective_action text, -- block, allow, re_authenticate
  created_at timestamp default now()
);
```

### 5. Session Behavioral Summary
```sql
create table session_behavioral_summary (
  id uuid primary key default uuid_generate_v4(),
  session_id uuid references sessions(id) not null,
  total_events integer default 0,
  unique_event_types text[],
  session_duration_seconds integer,
  total_vectors_generated integer default 0,
  average_confidence float default 0.0,
  anomaly_indicators text[],
  final_risk_assessment text, -- low, medium, high, critical
  created_at timestamp default now()
);
```

### 6. ML Model Metadata
```sql
create table ml_model_metadata (
  id uuid primary key default uuid_generate_v4(),
  model_name text not null,
  version text not null,
  model_type text not null, -- faiss, adaptive
  training_data_size integer,
  accuracy_metrics jsonb,
  deployment_date timestamp default now(),
  is_active boolean default true
);
```

## Run This SQL in Supabase

Execute the above SQL commands in your Supabase SQL editor to create the required tables.
