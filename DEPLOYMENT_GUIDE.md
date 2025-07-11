# BRIDGE ML-Engine Separate Deployment Guide

## Overview

This guide helps you deploy the BRIDGE ML-Engine as a separate service from your backend, enabling independent scaling and management of both components.

## Architecture

```
┌─────────────────────┐    HTTP API    ┌─────────────────────┐
│                     │   (REST/JSON)  │                     │
│   Backend Service   │◄──────────────►│  ML-Engine Service  │
│   (Port 8000)       │                │   (Port 8001)       │
│                     │                │                     │
└─────────────────────┘                └─────────────────────┘
```

## Prerequisites

### Backend Service
- Python 3.9+
- Your existing backend is already running
- Network access to ML-Engine service

### ML-Engine Service
- Python 3.9+
- 4GB+ RAM
- 2GB+ disk space
- CUDA 11.8+ (optional, for GPU acceleration)
- Docker (optional, for containerized deployment)

## Step-by-Step Deployment

### 1. Configure Backend for ML-Engine Connection

#### Update Backend Environment (.env)
```bash
# ML-Engine Configuration
ML_ENGINE_URL=http://localhost:8001
ML_ENGINE_ENABLED=true
ML_ENGINE_TIMEOUT=30
ML_ENGINE_API_KEY=your-secure-api-key-here
```

**For Production:**
```bash
# Use your actual ML-Engine service URL
ML_ENGINE_URL=https://ml-engine.your-domain.com
ML_ENGINE_ENABLED=true
ML_ENGINE_TIMEOUT=30
ML_ENGINE_API_KEY=your-production-api-key
```

#### Verify Backend Configuration
The backend already has the necessary files:
- ✅ `ml_engine_client.py` - HTTP client for ML-Engine
- ✅ `ml_hooks.py` - Integration hooks (now uses HTTP client)
- ✅ Session manager integration
- ✅ WebSocket event processing

### 2. Deploy ML-Engine Service

#### Option A: Local Development Deployment

**Windows:**
```cmd
cd bridge\ml-engine
start_ml_service.bat
```

**Linux/Mac:**
```bash
cd bridge/ml-engine
chmod +x start_ml_service.sh
./start_ml_service.sh
```

#### Option B: Docker Deployment

```bash
cd bridge/ml-engine

# Build and run with Docker Compose
docker-compose up -d

# Or build and run manually
docker build -t bridge-ml-engine .
docker run -d --name bridge-ml-engine -p 8001:8001 bridge-ml-engine
```

#### Option C: Production Server Deployment

```bash
# 1. Copy ML-Engine code to server
scp -r bridge/ml-engine user@your-server:/path/to/ml-engine

# 2. SSH to server and setup
ssh user@your-server
cd /path/to/ml-engine

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure environment
cp .env.example .env
# Edit .env with your configuration

# 6. Initialize models
python scripts/initialize_models.py

# 7. Start service
uvicorn ml_engine_api_service:app --host 0.0.0.0 --port 8001
```

### 3. Verify ML-Engine Service

#### Check Service Health
```bash
curl http://localhost:8001/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "ml_engine_initialized": true,
  "ml_engine_running": true
}
```

#### Check Service Statistics
```bash
curl http://localhost:8001/stats
```

### 4. Test Backend-to-ML-Engine Connection

#### Start Backend
```bash
cd bridge/backend
python main.py  # or uvicorn main:app --reload
```

#### Check Backend Logs
Look for these messages in backend logs:
```
INFO - ML-Engine hooks initialized - URL: http://localhost:8001, Enabled: true
INFO - ✅ ML-Engine connection established
```

#### Test Session Flow
1. Start a user session (login)
2. Generate some behavioral events
3. Check both backend and ML-Engine logs for successful communication

### 5. Production Considerations

#### Security
- Use HTTPS for ML-Engine service in production
- Implement API key authentication
- Use secure network connections (VPN/private network)

#### Monitoring
- Monitor ML-Engine health endpoint
- Set up alerts for service unavailability
- Track authentication decision metrics

#### Scaling
- Use load balancers for multiple ML-Engine instances
- Implement Redis for session state sharing
- Use container orchestration (Kubernetes)

## Configuration Reference

### Backend Configuration (.env)
```bash
# ML-Engine Service Connection
ML_ENGINE_URL=http://localhost:8001
ML_ENGINE_ENABLED=true
ML_ENGINE_TIMEOUT=30
ML_ENGINE_HEALTH_CHECK_INTERVAL=60
ML_ENGINE_API_KEY=your-api-key-here
```

### ML-Engine Configuration (.env)
```bash
# Environment
BRIDGE_ENV=production
ML_ENGINE_HOST=0.0.0.0
ML_ENGINE_PORT=8001
ML_ENGINE_WORKERS=1

# Model Configuration
MODEL_BASE_PATH=/app/models
FAISS_INDEX_PATH=/app/faiss/index
FAISS_DIMENSION=128

# Performance
MAX_CONCURRENT_SESSIONS=1000
INFERENCE_BATCH_SIZE=16
CACHE_SIZE=10000
MAX_WORKERS=8

# Security
ENCRYPTION_KEY_SIZE=256
VECTOR_ENCRYPTION=true
AUDIT_LOGGING=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=bridge_ml_engine.log
```

## API Endpoints

### ML-Engine Service Endpoints
- `GET /health` - Health check
- `GET /stats` - Service statistics
- `POST /session/start` - Start ML session
- `POST /session/process-event` - Process behavioral event
- `POST /session/end` - End ML session

### Backend API Endpoints (for monitoring)
- `GET /api/v1/ml/ml-engine/status` - Get ML-Engine status
- `GET /api/v1/ml/ml-engine/session/{session_id}` - Get ML session info

## Troubleshooting

### Common Issues

#### 1. "ML-Engine not available" in backend logs
- Check if ML-Engine service is running
- Verify ML_ENGINE_URL in backend .env
- Test connectivity: `curl http://ml-engine-url/health`

#### 2. Connection timeout errors
- Increase ML_ENGINE_TIMEOUT in backend .env
- Check network connectivity between services
- Verify firewall rules

#### 3. Authentication failures
- Ensure API keys match between services
- Check authentication configuration

#### 4. High latency
- Check ML-Engine resource usage
- Consider increasing ML_ENGINE_WORKERS
- Optimize model loading and caching

### Debugging Steps

1. **Check ML-Engine logs:**
   ```bash
   docker logs bridge-ml-engine
   # or check log files
   tail -f bridge_ml_engine.log
   ```

2. **Test ML-Engine directly:**
   ```bash
   curl -X POST http://localhost:8001/session/start \
     -H "Content-Type: application/json" \
     -d '{"session_id": "test", "user_id": "user123", "device_id": "device456", "phone": "1234567890"}'
   ```

3. **Check backend ML client:**
   ```python
   # In backend Python console
   from ml_engine_client import MLEngineClient
   client = MLEngineClient("http://localhost:8001")
   result = await client.health_check()
   print(f"ML-Engine available: {result}")
   ```

## Performance Monitoring

### Key Metrics to Monitor
- ML-Engine response time (<150ms target)
- Authentication accuracy (>95% target)
- Service availability (>99.5% target)
- Memory usage (<4GB target)
- Active sessions count

### Monitoring Setup
```bash
# Check service metrics
curl http://localhost:8001/stats

# Monitor with external tools
# - Prometheus + Grafana
# - New Relic / DataDog
# - AWS CloudWatch
```

## Backup and Recovery

### Data to Backup
- ML models and FAISS indices
- Configuration files
- Log files (for audit trails)

### Recovery Process
1. Restore ML models and indices
2. Restart ML-Engine service
3. Verify backend connectivity
4. Test authentication flow

## Support

For issues or questions:
- Check logs first (both backend and ML-Engine)
- Verify network connectivity
- Test each service independently
- Contact team for complex issues

---

**Ready to Deploy!** 🚀

Your BRIDGE ML-Engine is now ready for separate deployment. The backend will automatically connect to the ML-Engine service and provide continuous behavioral authentication for your banking application.
