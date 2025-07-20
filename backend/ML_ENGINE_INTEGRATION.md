# ML Engine Integration Guide

## Overview

This document explains how the backend integrates with the ML Engine for behavioral analysis and session management.

## Integration Points

### 1. Session Lifecycle Management

#### Session Start
- **Trigger**: When user verifies MPIN or uses MPIN-only login
- **Location**: `backend/app/api/v1/endpoints/auth.py`
- **Function**: `start_ml_session(user_id, session_id, device_info)`
- **Data Sent**: User ID, session ID, device information

```python
# In verify_mpin_endpoint and mpin_login
ml_result = await start_ml_session(user_id, session_id, device_info)
```

#### Session End
- **Trigger**: When user logs out, app closes, or WebSocket disconnects
- **Location**: Multiple files (auth.py, websocket.py, session_manager.py)
- **Function**: `end_ml_session(session_id, reason)`
- **Data Sent**: Session ID, termination reason

```python
# In logout, WebSocket disconnect, session termination
ml_end_result = await end_ml_session(session_id, "user_logout")
```

### 2. Behavioral Data Analysis

#### Real-time Analysis
- **Trigger**: When behavioral data is received via WebSocket
- **Location**: `backend/app/api/v1/endpoints/websocket.py`
- **Function**: `behavioral_event_hook(user_id, session_id, events)`
- **Data Sent**: User ID, session ID, behavioral events array

```python
# In process_behavioral_data
ml_result = await behavioral_event_hook(session.user_id, session_id, [behavioral_event])
```

#### ML Decision Handling
- **Decision Types**: "allow", "block", "monitor"
- **Confidence Threshold**: 0.8 for automatic blocking
- **Fallback**: Local rule-based analysis when ML engine is unavailable

```python
if decision == "block" and confidence > 0.8:
    session.block_session("ML Engine detected suspicious behavior")
```

### 3. App Lifecycle Events

#### New Endpoint
- **URL**: `POST /api/v1/ws/sessions/{session_id}/lifecycle`
- **Purpose**: Handle app state changes from frontend
- **Events**: app_close, app_background, app_foreground, user_logout

```python
# Frontend can call this when app state changes
{
    "event_type": "app_close",
    "details": {
        "reason": "user_exit",
        "timestamp": "2024-01-01T12:00:00Z"
    }
}
```

## Error Handling

### ML Engine Unavailability
- **Fallback Strategy**: Continue with local analysis
- **Logging**: All ML errors are logged as behavioral events
- **Graceful Degradation**: System continues to function

```python
except Exception as ml_error:
    session.add_behavioral_data("ml_session_error", {
        "session_id": session_id,
        "ml_engine_status": "error",
        "error": str(ml_error),
        "timestamp": session.last_activity.isoformat()
    })
```

### Connection Failures
- **Timeout**: 30 seconds for ML engine requests
- **Retry**: No automatic retry (to avoid blocking user experience)
- **Status Tracking**: ML engine availability is tracked

## Behavioral Data Flow

### 1. Data Collection
```
Frontend → WebSocket → Backend → ML Engine
```

### 2. Analysis Response
```
ML Engine → Backend → Session Risk Score Update
```

### 3. Decision Execution
```
Backend → Session Block/Allow → Frontend Notification
```

## API Endpoints

### Frontend-Facing Endpoints

#### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - Login with password
- `POST /api/v1/auth/verify-mpin` - MPIN verification (starts ML session)
- `POST /api/v1/auth/mpin-login` - MPIN-only login (starts ML session)
- `POST /api/v1/auth/logout` - Logout (ends ML session)

#### Behavioral Data
- `POST /api/v1/log/behavior-data` - Send behavioral events
- `WebSocket /api/v1/ws/behavior/{session_id}` - Real-time behavioral streaming

#### Session Management
- `POST /api/v1/ws/sessions/{session_id}/lifecycle` - App lifecycle events
- `GET /api/v1/auth/session-status` - Get session status

### Internal ML Engine Endpoints (Called by Backend)

- `POST /session/start` - Start ML session
- `POST /session/end` - End ML session
- `POST /analyze-mobile` - Analyze behavioral data
- `POST /feedback` - Submit feedback
- `GET /statistics` - Get ML engine statistics

## Configuration

### ML Engine Settings
```python
# In ml_engine_client.py
ML_ENGINE_URL = "http://127.0.0.1:8001"
TIMEOUT = 30.0  # seconds
```

### Risk Thresholds
```python
# In config.py
SUSPICIOUS_THRESHOLD = 0.7
HIGH_RISK_THRESHOLD = 0.9
```

## Testing

### Integration Test
Run the comprehensive integration test:

```bash
cd backend
python test_ml_integration.py
```

This test covers:
1. ML Engine health check
2. User registration and login
3. MPIN verification (ML session start)
4. Behavioral data transmission
5. App lifecycle events
6. Logout (ML session end)

### Manual Testing

#### Start Both Services
```bash
# Terminal 1 - Backend
cd backend
python main.py

# Terminal 2 - ML Engine
cd ml-engine
python main.py
```

#### Test Flow
1. Register user: `POST /api/v1/auth/register`
2. Login: `POST /api/v1/auth/login`
3. Verify MPIN: `POST /api/v1/auth/verify-mpin`
4. Send behavioral data: `POST /api/v1/log/behavior-data`
5. Test lifecycle: `POST /api/v1/ws/sessions/{id}/lifecycle`
6. Logout: `POST /api/v1/auth/logout`

## Monitoring

### Behavioral Events to Monitor
- `ml_session_started` - ML session successfully started
- `ml_session_failed` - ML session start failed
- `ml_session_error` - ML session error
- `ml_analysis_result` - ML analysis completed
- `ml_analysis_failed` - ML analysis failed
- `ml_session_ended` - ML session ended
- `ml_session_end_error` - ML session end error

### Log Locations
- Backend logs: Console output
- ML Engine logs: ML engine console
- Behavioral data: Session buffers and Supabase

## Troubleshooting

### Common Issues

#### 1. 500 Error on MPIN Verification
- **Cause**: ML Engine not running or unreachable
- **Solution**: Start ML Engine service
- **Fallback**: System continues with local analysis

#### 2. WebSocket Connection Issues
- **Cause**: Invalid session token or session not found
- **Solution**: Verify session token and session exists
- **Check**: Session status endpoint

#### 3. ML Engine Timeout
- **Cause**: ML Engine overloaded or network issues
- **Solution**: Check ML Engine logs and network
- **Fallback**: Local analysis continues

### Debug Commands

#### Check ML Engine Health
```bash
curl http://127.0.0.1:8001/
```

#### Check Backend Health
```bash
curl http://127.0.0.1:8000/health
```

#### Debug Session Token
```bash
curl http://127.0.0.1:8000/api/v1/ws/debug/token/{token}
```

## Security Considerations

### Data Privacy
- Behavioral data is encrypted in transit
- Session tokens have short expiration
- ML engine receives minimal required data

### Access Control
- All endpoints require valid authentication
- Session tokens are validated for each request
- ML engine endpoints are internal only

### Error Handling
- No sensitive data in error messages
- Graceful degradation when ML engine unavailable
- Comprehensive audit logging 