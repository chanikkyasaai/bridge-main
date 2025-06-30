# Canara AI Security Backend

A FastAPI-based backend service for ML-powered behavioral analysis and banking security.

## 🔐 Security Features

- **Session-Based Authentication**: JWT tokens with MPIN verification
- **Real-time Behavioral Analysis**: WebSocket-based data collection
- **ML-Powered Risk Assessment**: Continuous behavior pattern analysis
- **Adaptive Security**: Dynamic MPIN challenges and session blocking
- **Fraud Prevention**: Real-time session monitoring and risk scoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Flutter App   │◄──►│  FastAPI Backend │◄──►│   ML Engine     │
│                 │    │                  │    │                 │
│ • MPIN Entry    │    │ • Session Mgmt   │    │ • Risk Scoring  │
│ • WebSocket     │    │ • Behavioral     │    │ • Pattern       │
│ • Behavioral    │    │   Data Buffer    │    │   Analysis      │
│   Data Send     │    │ • Security Logic │    │ • Fraud Model   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Run setup script (Windows)
setup.bat

# Or manually:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

### 2. Start Server
```bash
# Use start script
start.bat

# Or manually:
python main.py
```

### 3. Test with Demo Client
```bash
python demo_client.py
```

## 📡 API Endpoints

### Authentication & Session Management
```
POST /api/v1/auth/register          # Register with email, password, MPIN
POST /api/v1/auth/login             # Login and create session
POST /api/v1/auth/verify-mpin       # Verify MPIN for current session
POST /api/v1/auth/mpin-challenge    # Handle MPIN challenge
GET  /api/v1/auth/session-status    # Get session status and risk score
POST /api/v1/auth/logout           # Logout and terminate session
```

### Behavioral Analysis WebSocket
```
WS /api/v1/ws/behavior/{session_id}?token={session_token}
```

### Session Analytics
```
GET /api/v1/ws/sessions/{session_id}/behavior-summary
POST /api/v1/ws/sessions/{session_id}/simulate-ml-analysis
```

## 🎯 User Flow

### 1. Registration
```json
POST /api/v1/auth/register
{
  "email": "user@canara.com",
  "password": "securepassword",
  "mpin": "1234"
}
```

### 2. Login & Session Creation
```json
POST /api/v1/auth/login
{
  "email": "user@canara.com",
  "password": "securepassword",
  "device_id": "mobile_app_001"
}

Response:
{
  "session_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "session_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

### 3. WebSocket Connection for Behavioral Data
```javascript
const ws = new WebSocket(
  `ws://localhost:8000/api/v1/ws/behavior/${sessionId}?token=${sessionToken}`
);

// Send behavioral events
ws.send(JSON.stringify({
  "event_type": "typing_pattern",
  "data": {
    "words_per_minute": 65,
    "key_press_intervals": [0.15, 0.12, 0.18, 0.14],
    "timestamp": "2025-06-24T10:30:00Z"
  }
}));
```

### 4. Handle Security Events
```javascript
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === "mpin_required") {
    // Show MPIN input dialog
    promptForMPIN();
  } else if (data.type === "session_blocked") {
    // Handle session block
    redirectToLogin();
  }
};
```

## 🧠 Behavioral Data Types

The system collects and analyzes various behavioral patterns:

### Typing Patterns
```json
{
  "event_type": "typing_pattern",
  "data": {
    "words_per_minute": 65,
    "key_press_intervals": [0.15, 0.12, 0.18],
    "backspace_frequency": 0.02
  }
}
```

### Navigation Behavior
```json
{
  "event_type": "navigation_pattern",
  "data": {
    "page_switches_per_minute": 4,
    "current_page": "transfer",
    "time_on_page": 45.2
  }
}
```

### Mouse/Touch Patterns
```json
{
  "event_type": "mouse_movement",
  "data": {
    "movement_speed": 120.5,
    "click_pattern": "normal",
    "idle_time": 2.1
  }
}
```

### Suspicious Indicators
```json
{
  "event_type": "rapid_clicks",
  "data": {
    "clicks_per_second": 12,
    "pattern": "automated"
  }
}
```

## ⚡ Security Logic

### Risk Scoring
- **0.0 - 0.4**: Normal behavior ✅
- **0.4 - 0.7**: Monitor closely 👀
- **0.7 - 0.9**: Request MPIN verification 🔐
- **0.9 - 1.0**: Block session immediately 🚫

### Security Actions
1. **MPIN Challenge**: When risk score ≥ 0.7
2. **Session Block**: When risk score ≥ 0.9
3. **Account Lock**: After 3 failed MPIN attempts

### Session Management
- Session expires after 60 minutes of inactivity
- Each session linked to specific device ID
- Behavioral data stored in session-specific buffer files
- Background cleanup of expired sessions

## 🔗 Integration with ML Engine

The backend is designed to integrate with the ML engine in `../ml-engine/`:

### Data Flow
1. **Real-time**: Behavioral data → Session buffer → WebSocket
2. **Batch Processing**: Buffer files → ML model → Risk scores
3. **Feedback Loop**: ML predictions → Session risk updates

### Buffer Files
```
session_buffers/
├── {session_id_1}.jsonl    # Behavioral events for session 1
├── {session_id_2}.jsonl    # Behavioral events for session 2
└── ...
```

Each line in buffer file:
```json
{
  "timestamp": "2025-06-24T10:30:00Z",
  "event_type": "typing_pattern",
  "data": {...},
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "user_email": "user@canara.com"
}
```

## 🧪 Testing

### Run Tests
```bash
pytest tests/
```

### Manual Testing with Demo Client
```bash
python demo_client.py
```

### WebSocket Testing
Use the demo client or any WebSocket testing tool to connect to:
```
ws://localhost:8000/api/v1/ws/behavior/{session_id}?token={session_token}
```

## 🚀 Deployment

### Environment Variables
```bash
# Security
SECRET_KEY=your-production-secret-key
MPIN_LENGTH=4
SUSPICIOUS_THRESHOLD=0.7
HIGH_RISK_THRESHOLD=0.9

# Session Management
SESSION_EXPIRE_MINUTES=60
BEHAVIOR_BUFFER_SIZE=1000

# Development
DEBUG=False
```

### Production Setup
1. Set strong `SECRET_KEY`
2. Configure proper CORS origins
3. Set up SSL/TLS for WebSocket connections
4. Implement proper logging and monitoring
5. Connect to actual database instead of mock data
6. Integrate with real ML model for risk scoring

## 📊 Monitoring

The system provides real-time monitoring of:
- Active sessions and their risk scores
- Behavioral event counts and patterns
- MPIN verification success/failure rates
- Session blocks and security events

Access monitoring at: `http://localhost:8000/docs`

## 🔮 Next Steps

1. **Database Integration**: Replace mock data with PostgreSQL
2. **ML Model Integration**: Connect with actual behavioral analysis models
3. **Advanced Analytics**: Implement pattern recognition algorithms
4. **Mobile SDK**: Create Flutter/React Native SDK for easy integration
5. **Admin Dashboard**: Build admin interface for monitoring and management

## 📜 License

This project is part of the Canara AI banking security solution.
