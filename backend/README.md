# Canara AI Security Backend - Supabase Integration

A FastAPI-based backend system for real-time behavioral analysis and security monitoring with Supabase integration, designed for banking applications.

## 🚀 Features

- **Real-time Behavioral Analysis**: WebSocket-based continuous data collection
- **Supabase Integration**: Database and storage using Supabase
- **Session Management**: Secure session handling with behavioral data buffering
- **MPIN Verification**: Multi-factor authentication with behavioral analysis
- **Structured Logging**: JSON-based behavioral logs stored in organized structure
- **ML-Ready**: Integration points for machine learning models
- **Security Events**: Comprehensive security event tracking

## 📊 Architecture

### Database Structure (Supabase)
```sql
-- Users table
users (
    id UUID PRIMARY KEY,
    phone TEXT UNIQUE,
    password_hash TEXT,
    mpin_hash TEXT,
    created_at TIMESTAMP
)

-- Sessions table
sessions (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    started_at TIMESTAMP,
    ended_at TIMESTAMP,
    device_info TEXT,
    is_escalated BOOLEAN,
    final_decision TEXT,
    anomaly_score FLOAT,
    session_token TEXT,
    log_file_url TEXT
)

-- Security events table
security_events (
    id SERIAL PRIMARY KEY,
    session_id UUID REFERENCES sessions(id),
    level INTEGER,
    decision TEXT,
    reason TEXT,
    timestamp TIMESTAMP,
    model_used TEXT,
    match_score FLOAT
)
```

### Storage Structure (Supabase Storage)
```
behavior-logs/
├── logs/
│   ├── {user_id}/
│   │   ├── {session_id_1}.json
│   │   ├── {session_id_2}.json
│   │   └── ...
│   └── ...
```

### JSON Log Format
```json
{
  "user_id": "uuid",
  "session_id": "uuid", 
  "uploaded_at": "timestamp",
  "total_events": 150,
  "logs": [
    {
      "timestamp": "2025-07-02T10:30:00Z",
      "event_type": "button_click",
      "data": {
        "button_id": "transfer_btn",
        "coordinates": [150, 200],
        "pressure": 0.8
      }
    }
  ]
}
```

## 🛠️ Setup

### 1. Environment Configuration

Create a `.env` file with your Supabase credentials:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_key_here
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Supabase Setup

Run the setup script to create required tables and storage:

```bash
python setup_supabase.py
```

Or manually create the tables using the SQL schema provided above.

### 4. Start the Backend

```bash
python main.py
```

The server will start on `http://localhost:8000`

## 📡 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user with phone/password/MPIN
- `POST /api/v1/auth/login` - Login and create session
- `POST /api/v1/auth/verify-mpin` - Verify MPIN for security challenges
- `POST /api/v1/auth/logout` - Logout and terminate session

### Behavioral Logging
- `POST /api/v1/log/start-session` - Start behavioral logging session
- `POST /api/v1/log/behavior-data` - Log behavioral data (stored in memory)
- `POST /api/v1/log/end-session` - End session and upload data to Supabase Storage
- `GET /api/v1/log/session/{id}/status` - Get session status and data summary
- `GET /api/v1/log/session/{id}/logs` - Retrieve logs from Supabase Storage

### WebSocket
- `ws://localhost:8000/api/v1/ws/behavior/{session_id}?token={token}` - Real-time behavioral data streaming

## 🔄 Workflow

### 1. User Registration & Login
```python
# Register
POST /auth/register
{
    "phone": "9876543210",
    "password": "SecurePassword123",
    "mpin": "123456"
}

# Login
POST /auth/login
{
    "phone": "9876543210", 
    "password": "SecurePassword123",
    "device_id": "device_001"
}
```

### 2. Start Behavioral Logging Session
```python
POST /log/start-session
{
    "user_id": "uuid",
    "phone": "9876543210",
    "device_id": "device_001",
    "device_info": "Android 12, Chrome 98"
}
```

### 3. Continuous Data Collection

**Option A: WebSocket (Recommended for real-time)**
```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/ws/behavior/session_id?token=jwt_token');

// Send behavioral events
ws.send(JSON.stringify({
    "event_type": "button_click",
    "data": {
        "button_id": "transfer_btn",
        "coordinates": [150, 200]
    }
}));
```

**Option B: REST API**
```python
POST /log/behavior-data
{
    "session_id": "session_uuid",
    "event_type": "typing_pattern",
    "data": {
        "field": "amount",
        "typing_speed": 45,
        "keystroke_dynamics": [0.1, 0.12, 0.08]
    }
}
```

### 4. End Session (Upload to Supabase)
```python
POST /log/end-session
{
    "session_id": "session_uuid",
    "final_decision": "normal"
}
```

This uploads all behavioral data to Supabase Storage as a structured JSON file.

## 🧪 Testing

Run the demo client to test the complete workflow:

```bash
python supabase_demo_client.py
```

This demonstrates:
- User registration and login
- Session creation with Supabase integration
- WebSocket behavioral data streaming
- REST API data logging
- Session termination with data upload
- Log retrieval from Supabase Storage

## 📊 Behavioral Events

The system supports various behavioral event types:

### User Interaction Events
- `button_click` - Button presses with coordinates and pressure
- `typing_pattern` - Keystroke dynamics and typing speed
- `mouse_movement` - Mouse movement patterns and velocity
- `page_view` - Page navigation and load times
- `form_interaction` - Form field interactions and focus patterns

### Security Events  
- `login_success/failure` - Authentication attempts
- `mpin_entry` - MPIN entry patterns and timing
- `security_check` - Device fingerprinting and verification
- `transaction_attempt` - Financial transaction patterns

### Session Events
- `idle_behavior` - User idle periods and resumption
- `navigation_pattern` - Application navigation patterns
- `risk_score_update` - ML model risk assessment updates

## 🔒 Security Features

- **Session Tokens**: JWT-based authentication
- **MPIN Verification**: 6-digit MPIN with attempt limiting
- **Risk Scoring**: Real-time behavioral risk assessment
- **Security Events**: Comprehensive audit trail
- **Data Encryption**: Secure data transmission and storage

## 🤖 ML Integration Points

The system provides integration points for machine learning models:

1. **Real-time Analysis**: Process behavioral events as they arrive
2. **Batch Analysis**: Analyze complete session data post-session
3. **Risk Scoring**: Update risk scores based on behavioral patterns
4. **Anomaly Detection**: Detect unusual behavioral patterns
5. **Security Decisions**: Automated security decisions (continue/re-auth/block)

## 📝 Configuration

Key settings in `app/core/config.py`:

```python
# Session Configuration
SESSION_EXPIRE_MINUTES = 60
BEHAVIOR_BUFFER_SIZE = 1000  # Events per session
SUSPICIOUS_THRESHOLD = 0.7   # Risk score threshold
HIGH_RISK_THRESHOLD = 0.9    # Auto-block threshold

# MPIN Configuration  
MPIN_LENGTH = 5
MAX_MPIN_ATTEMPTS = 3
MPIN_LOCKOUT_MINUTES = 15

# Supabase Configuration
SUPABASE_STORAGE_BUCKET = "behavior-logs"
```

## 🐛 Troubleshooting

### Common Issues

1. **Supabase Connection Error**
   - Verify SUPABASE_URL and SUPABASE_SERVICE_KEY in .env
   - Check network connectivity to Supabase

2. **Storage Upload Fails**  
   - Ensure 'behavior-logs' bucket exists
   - Verify bucket permissions allow uploads
   - Check service key has storage permissions

3. **WebSocket Connection Issues**
   - Verify session token is valid
   - Check firewall settings for WebSocket connections
   - Ensure session exists and is not blocked

4. **Database Table Errors**
   - Run setup_supabase.py to create tables
   - Verify SQL schema matches expected structure
   - Check foreign key constraints

## 🔄 Development

### Adding New Behavioral Events

1. Define event structure in your client application
2. Send via WebSocket or REST API with `event_type` and `data`
3. Add ML analysis logic in `websocket.py` if needed
4. Update risk scoring rules as appropriate

### Extending Database Schema

1. Add new tables/columns to Supabase
2. Update `supabase_client.py` with new methods
3. Add corresponding API endpoints
4. Update setup script with new schema

## 📈 Monitoring

- **Session Analytics**: Track active sessions and behavioral patterns
- **Security Events**: Monitor authentication failures and risk scores  
- **Storage Usage**: Monitor Supabase Storage usage and costs
- **API Performance**: Track endpoint response times and error rates

## 🚀 Production Deployment

For production deployment:

1. **Environment Variables**: Use secure environment variable management
2. **Database Security**: Enable RLS (Row Level Security) in Supabase
3. **API Rate Limiting**: Implement rate limiting for API endpoints
4. **Monitoring**: Set up comprehensive logging and monitoring
5. **Backup**: Configure automated database and storage backups
6. **SSL/TLS**: Ensure all connections use HTTPS/WSS
7. **Scaling**: Consider horizontal scaling for high-traffic scenarios

## 📄 License

This project is part of the Canara AI Security System.
