# Canara AI Backend - Quick Start Guide

## 🚀 Quick Setup (3 Steps)

### 1. Environment Setup
```bash
# Run the setup script
setup.bat

# Or manually:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Supabase
```bash
# Copy example environment file
copy .env.example .env

# Edit .env with your Supabase credentials:
# SUPABASE_URL=https://your-project.supabase.co
# SUPABASE_SERVICE_KEY=your_service_key_here

# Run Supabase setup
python setup_supabase.py
```

### 3. Start & Test
```bash
# Start the server
python main.py
# OR
start.bat

# Test with demo client
python supabase_demo_client.py
```

## 📁 Clean Project Structure

```
backend/
├── .env                    # Supabase credentials (create from .env.example)
├── .env.example           # Template for environment variables
├── main.py                # FastAPI application entry point
├── requirements.txt       # Python dependencies
├── setup.bat             # Windows setup script
├── start.bat             # Windows start script
├── setup_supabase.py     # Supabase database setup
├── supabase_demo_client.py # Demo client for testing
├── README.md             # Complete documentation
├── app/
│   ├── api/v1/
│   │   ├── api.py        # API router
│   │   └── endpoints/
│   │       ├── auth.py   # Authentication endpoints
│   │       ├── logging.py # Behavioral logging endpoints
│   │       └── websocket.py # WebSocket endpoints
│   └── core/
│       ├── config.py     # Application configuration
│       ├── security.py   # Security utilities
│       ├── session_manager.py # Session management
│       └── supabase_client.py # Supabase integration
└── tests/
    └── test_api.py       # API tests
```

## 🔗 Key Endpoints

- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/log/start-session` - Start behavioral logging
- `POST /api/v1/log/behavior-data` - Log behavioral data
- `POST /api/v1/log/end-session` - End session & upload to Supabase
- `ws://localhost:8000/api/v1/ws/behavior/{session_id}` - WebSocket

## 🔍 What Was Removed

The following unnecessary files/folders were removed during cleanup:
- `app/models/` - Old SQLAlchemy models (replaced with Supabase)
- `demo_client.py` - Old demo client (replaced with supabase_demo_client.py)
- `README_OLD.md` - Old README (replaced with Supabase documentation)
- `session_buffers/` - Local session files (now use Supabase Storage)

## 🎯 Core Features

1. **Real-time Behavioral Collection**: WebSocket-based continuous data streaming
2. **Memory Buffering**: Data stored in memory during active sessions
3. **Supabase Integration**: Database + Storage for complete data lifecycle
4. **Structured Logging**: JSON logs with user_id/session_id organization
5. **Security Events**: ML decisions and risk scoring integration
6. **Session Management**: Complete session lifecycle with cleanup

## 🐛 Troubleshooting

**Server won't start?**
- Check `.env` file exists with valid Supabase credentials
- Run `setup.bat` to install dependencies
- Verify Python 3.8+ is installed

**Demo client fails?**
- Ensure server is running on localhost:8000
- Check Supabase tables exist (run `setup_supabase.py`)
- Verify storage bucket 'behavior-logs' is created

**WebSocket connection fails?**
- Check firewall settings
- Verify session token is valid
- Ensure session exists and isn't blocked

## 🎉 Success Indicators

When everything is working correctly, you should see:
1. ✅ Server starts at http://localhost:8000
2. ✅ API docs available at http://localhost:8000/docs
3. ✅ Demo client completes all 10 steps successfully
4. ✅ Data appears in Supabase dashboard (users, sessions, security_events tables)
5. ✅ JSON logs appear in Supabase Storage (behavior-logs bucket)

## 📞 Support

- Check README.md for detailed documentation
- Review API documentation at /docs when server is running
- Test individual endpoints using the demo client as reference
