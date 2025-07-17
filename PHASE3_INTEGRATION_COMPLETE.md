# 🎯 Phase 3 API Integration - COMPLETE SETUP GUIDE

## 📋 Prerequisites

### 1. Supabase Database Setup
Execute the following SQL in your Supabase SQL editor:

```sql
-- User Profiles for ML Engine
create table user_profiles (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid references users(id) not null,
  current_session_count integer default 0,
  total_sessions integer default 0,
  current_phase text default 'learning',
  risk_score float default 0.0,
  last_activity timestamp default now(),
  behavioral_model_version integer default 1,
  created_at timestamp default now(),
  updated_at timestamp default now()
);

-- Behavioral Vectors Storage
create table behavioral_vectors (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid references users(id) not null,
  session_id uuid references sessions(id) not null,
  vector_data float[] not null,
  confidence_score float default 0.0,
  feature_source text not null,
  created_at timestamp default now()
);

-- Authentication Decisions
create table authentication_decisions (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid references users(id) not null,
  session_id uuid references sessions(id) not null,
  decision text not null,
  confidence float not null,
  similarity_score float,
  layer_used text not null,
  risk_factors text[],
  threshold_used float,
  processing_time_ms integer,
  created_at timestamp default now()
);

-- Behavioral Feedback
create table behavioral_feedback (
  id uuid primary key default uuid_generate_v4(),
  user_id uuid references users(id) not null,
  session_id uuid references sessions(id) not null,
  decision_id uuid references authentication_decisions(id) not null,
  was_correct boolean not null,
  feedback_source text not null,
  corrective_action text,
  created_at timestamp default now()
);

-- Session Behavioral Summary
create table session_behavioral_summary (
  id uuid primary key default uuid_generate_v4(),
  session_id uuid references sessions(id) not null,
  total_events integer default 0,
  unique_event_types text[],
  session_duration_seconds integer,
  total_vectors_generated integer default 0,
  average_confidence float default 0.0,
  anomaly_indicators text[],
  final_risk_assessment text,
  created_at timestamp default now()
);

-- ML Model Metadata
create table ml_model_metadata (
  id uuid primary key default uuid_generate_v4(),
  model_name text not null,
  version text not null,
  model_type text not null,
  training_data_size integer,
  accuracy_metrics jsonb,
  deployment_date timestamp default now(),
  is_active boolean default true
);

-- Create indexes for performance
create index idx_behavioral_vectors_user_id on behavioral_vectors(user_id);
create index idx_behavioral_vectors_session_id on behavioral_vectors(session_id);
create index idx_authentication_decisions_user_id on authentication_decisions(user_id);
create index idx_authentication_decisions_session_id on authentication_decisions(session_id);
```

## 🚀 Quick Start (3 Steps)

### Step 1: Start Complete System
```bash
# From the bridge directory
start_complete_system.bat
```

This will start both:
- **Backend API** on port 8000
- **ML Engine API** on port 8001

### Step 2: Verify System Health
Open in browser:
- Backend: http://localhost:8000/docs
- ML Engine: http://localhost:8001/docs
- Health Check: http://localhost:8000/api/v1/ml/health

### Step 3: Run Integration Test
```bash
# Install test dependencies
pip install websockets

# Run the complete integration test
python test_integration.py
```

## 📁 System Architecture

```
bridge/
├── backend/                    # Main Backend (Port 8000)
│   ├── app/
│   │   ├── ml_engine_client.py    # HTTP client for ML Engine
│   │   ├── ml_hooks.py            # Integration hooks
│   │   └── api/v1/endpoints/
│   │       ├── ml_engine.py       # ML admin endpoints
│   │       ├── auth.py            # Updated with ML integration
│   │       └── websocket.py       # Updated with ML analysis
│   └── requirements.txt           # Updated dependencies
│
├── behavioral-auth-engine/     # ML Engine (Port 8001)
│   ├── ml_engine_api_service.py   # FastAPI ML service
│   ├── src/                       # Phase 1 & 2 components
│   │   ├── layers/                # FAISS & Adaptive layers
│   │   ├── core/                  # Vector store, session mgmt
│   │   └── data/                  # Models
│   ├── requirements.txt           # ML dependencies
│   └── start_ml_engine.bat        # ML Engine startup
│
├── start_complete_system.bat   # Start both systems
└── test_integration.py         # Complete integration test
```

## 🔄 Integration Flow

### 1. Session Lifecycle
```
User Login → MPIN Verification → START ML SESSION → Behavioral Collection → END ML SESSION → Logout
```

### 2. Real-time Analysis
```
WebSocket Event → ML Analysis → Authentication Decision → Action (Allow/Challenge/Block)
```

### 3. Feedback Loop
```
User Feedback → ML Model Adaptation → Improved Decisions
```

## 🎯 API Endpoints

### Backend Endpoints (Port 8000)
- `POST /api/v1/auth/verify-mpin` - Starts ML session
- `POST /api/v1/auth/logout` - Ends ML session  
- `WS /api/v1/ws/behavior/{session_id}` - Real-time behavioral data
- `GET /api/v1/ml/health` - ML Engine health check
- `POST /api/v1/ml/feedback` - Submit feedback
- `GET /api/v1/ml/status` - ML Engine status (auth required)

### ML Engine Endpoints (Port 8001)
- `GET /` - Health check
- `POST /session/start` - Start ML session
- `POST /session/end` - End ML session
- `POST /analyze` - Analyze behavioral data
- `POST /feedback` - Submit feedback for learning
- `GET /statistics` - ML Engine statistics

## 🧪 Testing Guide

### 1. Manual Testing
1. Start system: `start_complete_system.bat`
2. Open backend docs: http://localhost:8000/docs
3. Test authentication endpoints
4. Check ML integration status

### 2. Automated Testing
```bash
python test_integration.py
```

### 3. WebSocket Testing
Use the WebSocket endpoint with a valid session token:
```
ws://localhost:8000/api/v1/ws/behavior/{session_id}?token={session_token}
```

## 🔧 Configuration

### Backend Configuration (.env)
```env
# Existing Supabase config
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_key

# Optional: ML Engine URL (default: http://127.0.0.1:8001)
ML_ENGINE_URL=http://127.0.0.1:8001
```

### ML Engine Configuration
The ML Engine uses the behavioral-auth-engine configuration from Phase 1 & 2.

## 🚨 Troubleshooting

### ML Engine Not Available
- Check if ML Engine is running on port 8001
- Backend will work with fallback decisions when ML is unavailable
- Check logs in the ML Engine terminal window

### WebSocket Connection Issues
- Ensure valid session token
- Check session is active
- Verify WebSocket URL format

### Database Issues
- Ensure all new tables are created in Supabase
- Check database connection in both services
- Verify table permissions

## 📊 Monitoring

### Health Checks
- Backend health: http://localhost:8000/api/v1/ml/health
- ML Engine health: http://localhost:8001/
- System metrics: http://localhost:8000/api/v1/ml/metrics

### Logs
- Backend logs: Console window
- ML Engine logs: ML Engine window
- Integration test logs: Console output

## 🎉 Success Criteria

✅ Both services start without errors  
✅ Health checks return "healthy" status  
✅ Integration test passes all steps  
✅ WebSocket connections work  
✅ ML analysis produces decisions  
✅ Feedback submission works  
✅ Session lifecycle completes properly  

## 🔄 Next Steps

1. **Production Deployment**: Configure for production environment
2. **Monitoring**: Add comprehensive logging and metrics
3. **Security**: Implement additional security measures
4. **Scaling**: Configure for multiple instances
5. **UI Integration**: Connect with frontend application

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review console logs for error messages
3. Ensure all prerequisites are met
4. Test individual components separately
