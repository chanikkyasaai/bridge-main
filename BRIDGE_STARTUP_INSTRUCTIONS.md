# BRIDGE System Startup Instructions

## Overview
The BRIDGE banking security system consists of two main services that must run in parallel:

1. **ML-Engine Service** (Port 8001) - Handles behavioral verification and machine learning
2. **Backend Service** (Port 8000) - Handles API endpoints, session management, and user interfaces

## Prerequisites

### Python Environment
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Required Dependencies
Both services will automatically install their dependencies, but you can manually install:

```bash
# In ml-engine directory
pip install fastapi uvicorn torch transformers scikit-learn faiss-cpu numpy pandas

# In backend directory  
pip install fastapi uvicorn pydantic supabase aiohttp websockets
```

## Quick Start (Recommended)

### Option 1: Automated Startup
1. Open Command Prompt as Administrator (recommended)
2. Navigate to the project root: `cd C:\Users\Hp\OneDrive\Desktop\bridge\bridge`
3. Run the startup script: `start_bridge_system.bat`
4. Two command windows will open - one for each service

### Option 2: Manual Startup (for debugging)

#### Start ML-Engine First:
1. Open Command Prompt
2. Navigate to ML-Engine: `cd C:\Users\Hp\OneDrive\Desktop\bridge\bridge\ml-engine`
3. Run: `start_ml_engine.bat`
4. Wait for "ML-Engine initialized successfully" message

#### Start Backend Second:
1. Open another Command Prompt
2. Navigate to Backend: `cd C:\Users\Hp\OneDrive\Desktop\bridge\bridge\backend`
3. Run: `start_backend.bat`
4. Wait for "ML-Engine integration initialized successfully" message

## Service URLs

- **ML-Engine API**: http://localhost:8001
- **Backend API**: http://localhost:8000
- **Health Checks**: 
  - ML-Engine: http://localhost:8001/health
  - Backend: http://localhost:8000/health

## Session Lifecycle Integration

### When a Session Starts:
1. Frontend/App calls Backend `/auth/login` or similar endpoint
2. Backend validates credentials and creates session
3. Backend calls ML-Engine `/session/start` with session context
4. ML-Engine initializes behavioral monitoring for the session
5. Both services are now tracking the session

### During Session (Behavioral Monitoring):
1. Frontend sends behavioral events to Backend WebSocket
2. Backend processes and forwards events to ML-Engine `/session/verify`
3. ML-Engine analyzes behavioral patterns using Layer 1 (FAISS) and Layer 2 (Adaptive Context)
4. ML-Engine returns verification decision (allow/monitor/challenge/block)
5. Backend acts on the decision (continue session, request re-auth, block user, etc.)

### When Session Ends:
1. User logs out or session expires
2. Backend calls ML-Engine `/session/end`
3. ML-Engine cleans up session data and provides session summary
4. Both services clean up session-related resources

## Service Communication

The Backend communicates with ML-Engine via HTTP API calls:

- **Session Start**: `POST /session/start`
- **Behavioral Verification**: `POST /session/verify` 
- **Session End**: `POST /session/end`
- **Health Check**: `GET /health`
- **Statistics**: `GET /stats`

## Environment Variables

### ML-Engine Service:
- `ML_ENGINE_HOST=0.0.0.0`
- `ML_ENGINE_PORT=8001`
- `ML_ENGINE_WORKERS=1`

### Backend Service:
- `BACKEND_HOST=0.0.0.0`
- `BACKEND_PORT=8000`
- `ML_ENGINE_URL=http://localhost:8001`
- `ML_ENGINE_ENABLED=true`

## Troubleshooting

### ML-Engine Won't Start:
1. Check Python version: `python --version`
2. Install missing packages: `pip install -r requirements.txt`
3. Check port 8001 is not in use: `netstat -an | findstr :8001`

### Backend Can't Connect to ML-Engine:
1. Ensure ML-Engine is running first
2. Check ML-Engine health: http://localhost:8001/health
3. Verify ML_ENGINE_URL in backend environment

### Performance Issues:
1. Monitor CPU/memory usage in both command windows
2. Check logs for errors or warnings
3. Reduce ML_ENGINE_WORKERS if system is overloaded

### Ports Already in Use:
1. Change ports in the startup scripts
2. Update ML_ENGINE_URL in backend accordingly
3. Update firewall rules if needed

## Stopping Services

### Safe Shutdown:
1. Close frontend/app connections first
2. Press Ctrl+C in Backend command window
3. Press Ctrl+C in ML-Engine command window
4. Or close the command windows

### Force Stop:
1. Task Manager → Find python.exe processes
2. End processes related to uvicorn/fastapi

## Development Mode

For development, you can run services with auto-reload:

```bash
# ML-Engine (in ml-engine directory)
python ml_engine_api_service.py

# Backend (in backend directory)  
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Testing the Integration

1. Start both services
2. Check health endpoints:
   ```bash
   curl http://localhost:8001/health
   curl http://localhost:8000/health
   ```
3. Test session lifecycle:
   ```bash
   # Start session
   curl -X POST http://localhost:8001/session/start \
     -H "Content-Type: application/json" \
     -d '{"session_id":"test123","user_id":"user1","phone":"1234567890","device_id":"dev1","context":{}}'
   
   # End session
   curl -X POST http://localhost:8001/session/end \
     -H "Content-Type: application/json" \
     -d '{"session_id":"test123","user_id":"user1"}'
   ```

## Production Considerations

1. **Security**: Change default ports, enable HTTPS, configure firewalls
2. **Scaling**: Use multiple workers, load balancing, container orchestration
3. **Monitoring**: Set up logging, metrics collection, alerting
4. **Database**: Configure persistent storage for ML models and session data
5. **Backup**: Regular backups of ML models and configuration

## Support

For issues:
1. Check service logs in command windows
2. Verify all dependencies are installed
3. Ensure services can communicate (network/firewall)
4. Check Python version compatibility
