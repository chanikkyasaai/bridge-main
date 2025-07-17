# 🎯 Behavioral Authentication Frontend Integration Guide

## ✅ INTEGRATION STATUS: COMPLETE

The behavioral authentication engine is **fully integrated** and operational! Here's exactly how your frontend team should connect to it.

## 🔄 Complete Integration Flow

### Step 1: User Authentication
```javascript
// Frontend login with MPIN
const loginResponse = await fetch('http://localhost:8000/api/v1/auth/mpin-login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        phone: '9876543210',
        mpin: '123456',
        device_id: 'user_device_001'
    })
});

const authData = await loginResponse.json();
// Extract: session_id, session_token, access_token
```

### Step 2: WebSocket Behavioral Connection
```javascript
// Connect to behavioral streaming WebSocket
const wsUrl = `ws://localhost:8000/api/v1/behavior/${authData.session_id}?token=${authData.session_token}`;
const behavioralWS = new WebSocket(wsUrl);

behavioralWS.onopen = () => {
    console.log('Behavioral monitoring started');
};

behavioralWS.onmessage = (event) => {
    const response = JSON.parse(event.data);
    
    switch(response.type) {
        case 'connection_established':
            console.log('Behavioral streaming active');
            break;
            
        case 'mpin_required':
            // 🔐 SHOW MPIN RE-AUTHENTICATION DIALOG
            showMPINDialog();
            break;
            
        case 'session_blocked':
            // 🚨 FORCE LOGOUT - HIGH RISK DETECTED
            forceLogout();
            break;
            
        case 'data_received':
            // ✅ Behavioral data processed successfully
            break;
    }
};
```

### Step 3: Stream Behavioral Data
```javascript
// Stream user behavioral events
function streamBehavior(eventType, data) {
    if (behavioralWS.readyState === WebSocket.OPEN) {
        behavioralWS.send(JSON.stringify({
            event_type: eventType,
            data: data,
            timestamp: new Date().toISOString()
        }));
    }
}

// Example behavioral events:
streamBehavior('typing_pattern', {
    typing_speed: 65,
    keystroke_intervals: [110, 120, 105],
    accuracy: 0.95
});

streamBehavior('touch_behavior', {
    touch_pressure: [0.7, 0.8],
    touch_duration: [150, 140],
    coordinates: [{x: 100, y: 200}]
});

streamBehavior('navigation_pattern', {
    page_switches_per_minute: 2,
    dwell_time: 45,
    scroll_behavior: 'normal'
});
```

### Step 4: Handle Risk-Based Actions
```javascript
function showMPINDialog() {
    // Show MPIN input dialog to user
    const userMPIN = prompt('Please re-enter your MPIN for security verification:');
    
    // Send MPIN verification
    streamBehavior('mpin_verified', {
        mpin: userMPIN,
        verification_time: new Date().toISOString(),
        biometric_match: true
    });
}

function forceLogout() {
    // Close behavioral WebSocket
    behavioralWS.close();
    
    // Clear session data
    localStorage.clear();
    sessionStorage.clear();
    
    // Redirect to login
    window.location.href = '/login';
    
    // Show security message
    alert('Session blocked due to suspicious activity. Please login again.');
}
```

## 🔧 Technical Implementation Details

### Risk Thresholds (Automatically Configured):
- **Suspicious (0.7)**: Triggers MPIN re-authentication
- **High Risk (0.9)**: Blocks session and forces logout

### Key Behavioral Events to Monitor:
- **Typing Patterns**: Speed, rhythm, accuracy, keystroke timing
- **Touch Behavior**: Pressure, duration, coordinates, gestures
- **Navigation Patterns**: Page flow, dwell time, back button usage
- **Transaction Context**: Amount, time, location, beneficiary type

### WebSocket Message Types:
- `connection_established`: Behavioral monitoring started
- `data_received`: Event processed successfully
- `mpin_required`: User must re-authenticate with MPIN
- `session_blocked`: High risk detected, session terminated

## 🌟 System Features

### ✅ What's Working Right Now:
1. **Real-time Behavioral Streaming**: WebSocket endpoint active on `/api/v1/behavior/{session_id}`
2. **ML Risk Assessment**: Phase 1 + Phase 2 analysis with FAISS and adaptive learning
3. **Automatic Security Actions**: MPIN prompts and session blocking based on risk
4. **Session Management**: Complete lifecycle with behavioral tracking
5. **Database Integration**: Supabase with 5 tables for behavioral data persistence

### 🔄 Continuous User Verification:
- Every user action is analyzed in real-time
- Risk score updated continuously
- Immediate response to suspicious patterns
- Seamless security without user friction

## 🚀 Production Deployment

### Backend Services:
- **Backend API**: Running on port 8000
- **ML Engine**: Running on port 8001
- **Database**: Supabase PostgreSQL configured

### Frontend Requirements:
- WebSocket support for real-time communication
- MPIN dialog component for re-authentication
- Session management for risk-based logout
- Behavioral event collection from user interactions

## 📱 Mobile App Integration

For your Canara AI Flutter app:
```dart
// WebSocket connection
WebSocketChannel channel = WebSocketChannel.connect(
  Uri.parse('ws://localhost:8000/api/v1/behavior/$sessionId?token=$sessionToken'),
);

// Stream behavioral data
channel.sink.add(jsonEncode({
  'event_type': 'touch_behavior',
  'data': {
    'touch_pressure': touchPressure,
    'touch_duration': touchDuration,
    'coordinates': touchCoordinates,
  }
}));

// Handle risk responses
channel.stream.listen((data) {
  final response = jsonDecode(data);
  
  switch(response['type']) {
    case 'mpin_required':
      showMPINDialog();
      break;
    case 'session_blocked':
      forceLogout();
      break;
  }
});
```

## 🎯 Integration Complete!

Your behavioral authentication system is **100% ready** for frontend integration. The complete flow from user login through real-time behavioral analysis to security actions is operational.

**Next Steps:**
1. Connect your frontend to the WebSocket endpoint
2. Implement behavioral data collection
3. Handle MPIN re-auth and session blocking
4. Test with real user interactions

The backend will automatically handle risk assessment, ML analysis, and trigger appropriate security responses! 🚀✨
