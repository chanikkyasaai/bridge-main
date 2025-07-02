# 🚀 Frontend Integration API Routes

## Base URL
```
http://localhost:8000
```

## 📋 Complete API Route List

### 🏠 **System Routes**

#### 1. Root Information
```http
GET /
```
**Response:** System information, available endpoints, and features

#### 2. Health Check
```http
GET /health
```
**Response:** 
```json
{
  "status": "healthy",
  "service": "canara-ai-backend"
}
```

---

## 🔐 **Authentication Routes** (`/api/v1/auth`)

### 1. User Registration
```http
POST /api/v1/auth/register
```
**Request Body:**
```json
{
  "phone": "9876543210",
  "password": "SecurePassword123",
  "mpin": "123456"
}
```
**Response:**
```json
{
  "message": "User registered successfully",
  "user_id": "uuid-string",
  "next_step": "verify_otp_or_login"
}
```

### 2. User Login
```http
POST /api/v1/auth/login
```
**Request Body:**
```json
{
  "phone": "9876543210",
  "password": "SecurePassword123",
  "device_id": "device_unique_id"
}
```
**Response:**
```json
{
  "access_token": "jwt_access_token",
  "refresh_token": "jwt_refresh_token",
  "token_type": "bearer",
  "expires_in": 900
}
```

**Note:** Login returns only authentication tokens. Behavioral session starts only after MPIN verification.

### 3. Token Refresh
```http
POST /api/v1/auth/refresh
```
**Request Body:**
```json
{
  "refresh_token": "existing_refresh_token"
}
```
**Response:** Same as login response with refreshed tokens

### 4. MPIN-Only Login (Returning Users)
```http
POST /api/v1/auth/mpin-login
```
**Request Body:**
```json
{
  "phone": "9876543210",
  "mpin": "123456",
  "device_id": "device_unique_id"
}
```
**Response:**
```json
{
  "access_token": "jwt_access_token",
  "refresh_token": "jwt_refresh_token",
  "token_type": "bearer",
  "expires_in": 900,
  "session_id": "session_uuid",
  "session_token": "session_jwt_token",
  "behavioral_logging": "started",
  "message": "MPIN login successful - behavioral logging started"
}
```

**Note:** This endpoint is perfect for returning users who just need to enter MPIN to access the app. It creates both authentication tokens AND starts behavioral session in one step.

### 5. MPIN Verification (For Already Logged-in Users)
```http
POST /api/v1/auth/verify-mpin
```
**Headers:**
```http
Authorization: Bearer <access_token>
```
**Request Body:**
```json
{
  "mpin": "123456"
}
```
**Response:**
```json
{
  "message": "MPIN verified successfully",
  "user_id": "user_uuid",
  "status": "verified",
  "session_id": "session_uuid",
  "session_token": "session_jwt_token",
  "behavioral_logging": "started"
}
```

**Note:** Behavioral session starts here after successful MPIN verification.

### 6. User Logout
```http
POST /api/v1/auth/logout
```
**Headers:**
```http
Authorization: Bearer <access_token>
```
**Response:**
```json
{
  "message": "Logged out successfully"
}
```

### 7. Session Status
```http
GET /api/v1/auth/session-status
```
**Headers:**
```http
Authorization: Bearer <access_token>
```
**Response:**
```json
{
  "user_id": "user_uuid",
  "phone": "9876543210",
  "device_id": "device_id",
  "session_active": true,
  "login_time": "2025-07-02T10:30:00Z"
}
```

### 8. Get Active Tokens
```http
GET /api/v1/auth/tokens/active
```
**Headers:**
```http
Authorization: Bearer <access_token>
```
**Response:**
```json
{
  "active_tokens": [
    {
      "device_id": "device_1",
      "created_at": "2025-07-02T10:30:00Z",
      "expires_at": "2025-08-01T10:30:00Z"
    }
  ]
}
```

### 9. Revoke Device Tokens
```http
POST /api/v1/auth/tokens/revoke-device
```
**Headers:**
```http
Authorization: Bearer <access_token>
```
**Request Body:**
```json
{
  "device_id": "device_to_revoke"
}
```

---

## 📊 **Behavioral Logging Routes** (`/api/v1/log`)

### 1. Start Session
```http
POST /api/v1/log/start-session
```
**Request Body:**
```json
{
  "user_id": "user_uuid",
  "phone": "9876543210",
  "device_id": "device_unique_id",
  "device_info": "iPhone 13 Pro"
}
```
**Response:**
```json
{
  "session_id": "session_uuid",
  "supabase_session_id": "supabase_uuid",
  "message": "Session started successfully",
  "status": "active"
}
```

### 2. Log Behavioral Data
```http
POST /api/v1/log/behavior-data
```
**Headers:**
```http
Authorization: Bearer <session_token>
```
**Request Body:**
```json
{
  "session_id": "session_uuid",
  "event_type": "mouse_movement",
  "data": {
    "x": 150,
    "y": 200,
    "timestamp": "2025-07-02T10:30:00Z",
    "velocity": 0.5
  }
}
```
**Response:**
```json
{
  "message": "Behavioral data logged successfully",
  "session_id": "session_uuid",
  "event_type": "mouse_movement",
  "total_events": 25,
  "timestamp": "2025-07-02T10:30:00Z"
}
```

### 3. End Session
```http
POST /api/v1/log/end-session
```
**Headers:**
```http
Authorization: Bearer <session_token>
```
**Request Body:**
```json
{
  "session_id": "session_uuid",
  "final_decision": "transaction_completed"
}
```
**Response:**
```json
{
  "message": "Session ended successfully",
  "session_id": "session_uuid",
  "final_decision": "transaction_completed",
  "behavioral_data_saved": true,
  "timestamp": "2025-07-02T10:30:00Z"
}
```

### 4. Get Session Status
```http
GET /api/v1/log/session/{session_id}/status
```
**Headers:**
```http
Authorization: Bearer <session_token>
```
**Response:**
```json
{
  "session_id": "session_uuid",
  "user_id": "user_uuid",
  "phone": "9876543210",
  "device_id": "device_id",
  "is_active": true,
  "is_blocked": false,
  "risk_score": 0.35,
  "created_at": "2025-07-02T10:30:00Z",
  "last_activity": "2025-07-02T10:35:00Z",
  "behavioral_data_summary": {
    "total_events": 45,
    "event_types": ["mouse_movement", "key_press", "navigation"],
    "last_event": "2025-07-02T10:35:00Z"
  }
}
```

### 5. Get Session Logs
```http
GET /api/v1/log/session/{session_id}/logs
```
**Headers:**
```http
Authorization: Bearer <session_token>
```
**Response:**
```json
{
  "session_id": "session_uuid",
  "logs": [
    {
      "timestamp": "2025-07-02T10:30:00Z",
      "event_type": "mouse_movement",
      "data": { "x": 150, "y": 200 }
    }
  ],
  "file_path": "logs/user_id/session_id.json"
}
```

---

## 🔗 **WebSocket Routes** (`/api/v1/ws`)

### 1. Real-Time Behavioral Data
```websocket
ws://localhost:8000/api/v1/ws/behavior/{session_id}?token={session_token}
```

**Connect:**
```javascript
const websocket = new WebSocket(
  `ws://localhost:8000/api/v1/ws/behavior/${sessionId}?token=${sessionToken}`
);
```

**Send Data:**
```json
{
  "event_type": "mouse_click",
  "data": {
    "x": 100,
    "y": 50,
    "button": "left",
    "timestamp": "2025-07-02T10:30:00Z"
  }
}
```

**Receive Responses:**
```json
{
  "type": "connection_established",
  "session_id": "session_uuid",
  "message": "Behavioral data collection started",
  "timestamp": "2025-07-02T10:30:00Z"
}
```

```json
{
  "type": "data_received",
  "status": "processed",
  "timestamp": "2025-07-02T10:30:00Z"
}
```

### 2. Get Behavior Summary
```http
GET /api/v1/ws/sessions/{session_id}/behavior-summary
```
**Response:**
```json
{
  "session_id": "session_uuid",
  "risk_score": 0.25,
  "total_events": 150,
  "event_breakdown": {
    "mouse_movement": 45,
    "key_press": 30,
    "navigation": 20
  },
  "session_duration_minutes": 5.5,
  "last_activity": "2025-07-02T10:35:00Z",
  "is_blocked": false
}
```

### 3. Simulate ML Analysis
```http
POST /api/v1/ws/sessions/{session_id}/simulate-ml-analysis
```
**Response:**
```json
{
  "message": "ML analysis simulation completed",
  "session_id": "session_uuid",
  "predicted_risk_score": 0.75,
  "action_taken": "monitor"
}
```

---

## 🎯 **Frontend Integration Examples**

### JavaScript/TypeScript Example:

```javascript
class CanaraAIClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
    this.accessToken = null;
    this.sessionToken = null;
    this.sessionId = null;
  }

  // Authentication
  async register(phone, password, mpin) {
    const response = await fetch(`${this.baseUrl}/api/v1/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phone, password, mpin })
    });
    return response.json();
  }

  async login(phone, password, deviceId) {
    const response = await fetch(`${this.baseUrl}/api/v1/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phone, password, device_id: deviceId })
    });
    const data = await response.json();
    
    if (response.ok) {
      this.accessToken = data.access_token;
      this.refreshToken = data.refresh_token;
      // Note: No session tokens yet - they come after MPIN verification
    }
    return data;
  }

  async mpinLogin(phone, mpin, deviceId) {
    const response = await fetch(`${this.baseUrl}/api/v1/auth/mpin-login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ phone, mpin, device_id: deviceId })
    });
    const data = await response.json();
    
    if (response.ok) {
      // Get both authentication AND session tokens in one step!
      this.accessToken = data.access_token;
      this.refreshToken = data.refresh_token;
      this.sessionToken = data.session_token;
      this.sessionId = data.session_id;
      // Behavioral session starts immediately!
    }
    return data;
  }

  async verifyMPIN(mpin) {
    const response = await fetch(`${this.baseUrl}/api/v1/auth/verify-mpin`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.accessToken}`
      },
      body: JSON.stringify({ mpin })
    });
    const data = await response.json();
    
    if (response.ok) {
      this.sessionToken = data.session_token;
      this.sessionId = data.session_id;
      // Behavioral session starts here!
    }
    return data;
  }

  // Behavioral Logging
  async startSession(userId, phone, deviceId, deviceInfo) {
    const response = await fetch(`${this.baseUrl}/api/v1/log/start-session`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: userId, phone, device_id: deviceId, device_info: deviceInfo })
    });
    const data = await response.json();
    
    if (response.ok) {
      this.sessionId = data.session_id;
    }
    return data;
  }

  async logBehavior(eventType, eventData) {
    const response = await fetch(`${this.baseUrl}/api/v1/log/behavior-data`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.sessionToken}`
      },
      body: JSON.stringify({
        session_id: this.sessionId,
        event_type: eventType,
        data: eventData
      })
    });
    return response.json();
  }

  async endSession(finalDecision = 'normal') {
    const response = await fetch(`${this.baseUrl}/api/v1/log/end-session`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.sessionToken}`
      },
      body: JSON.stringify({
        session_id: this.sessionId,
        final_decision: finalDecision
      })
    });
    return response.json();
  }

  // WebSocket Connection
  connectWebSocket() {
    const wsUrl = `ws://localhost:8000/api/v1/ws/behavior/${this.sessionId}?token=${this.sessionToken}`;
    this.websocket = new WebSocket(wsUrl);
    
    this.websocket.onopen = () => {
      console.log('WebSocket connected');
    };
    
    this.websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('WebSocket message:', data);
    };
    
    this.websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    
    return this.websocket;
  }

  sendBehaviorData(eventType, data) {
    if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
      this.websocket.send(JSON.stringify({ event_type: eventType, data }));
    }
  }
}

// Usage Examples
const client = new CanaraAIClient();

// === FIRST-TIME USER WORKFLOW ===
// Step 1: Register new user
await client.register('9876543210', 'password123', '123456');

// Step 2: Login (authentication only)
await client.login('9876543210', 'password123', 'device001');

// Step 3: Verify MPIN (starts behavioral session)
const mpinResult = await client.verifyMPIN('123456');

// === RETURNING USER WORKFLOW (RECOMMENDED) ===
// Single step: MPIN login (authentication + session in one call)
await client.mpinLogin('9876543210', '123456', 'device001');

// === COMMON STEPS FOR BOTH WORKFLOWS ===
// Connect WebSocket for real-time data (session is now active)
client.connectWebSocket();

// Log behavioral events
client.sendBehaviorData('mouse_movement', { x: 100, y: 200 });
await client.logBehavior('page_navigation', { from: 'dashboard', to: 'transfer' });

// End session when user closes app
await client.endSession('app_closed');
```

---

## 🔄 **Common Behavioral Event Types**

### Mouse Events:
- `mouse_movement` - Mouse position and velocity
- `mouse_click` - Click events with coordinates
- `mouse_scroll` - Scroll behavior patterns

### Keyboard Events:
- `key_press` - Individual key presses
- `typing_pattern` - Typing speed and rhythm
- `copy_paste` - Copy/paste behavior

### Navigation Events:
- `page_navigation` - Page transitions
- `form_interaction` - Form field interactions
- `button_click` - UI element interactions

### Authentication Events:
- `login_attempt` - Login behavior
- `mpin_entry` - MPIN input patterns
- `biometric_verification` - Biometric auth events

### Transaction Events:
- `transaction_start` - Transaction initiation
- `amount_entry` - Amount input behavior
- `beneficiary_selection` - Recipient selection

---

## 🚨 **Error Handling**

All endpoints return appropriate HTTP status codes:
- `200` - Success
- `400` - Bad Request (validation errors)
- `401` - Unauthorized (invalid/expired tokens)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found (resource doesn't exist)
- `500` - Internal Server Error

Example error response:
```json
{
  "detail": "Invalid phone number format"
}
```

---

## 🔒 **Authentication Flow for Frontend**

### 📱 **First-Time Users (New Installation):**
1. **User Registration** → Account created in Supabase
2. **User Login** → Get `access_token` and `refresh_token` (no session yet)
3. **User Enters MPIN** → **BEHAVIORAL SESSION STARTS** → Get `session_id` and `session_token`
4. **Real-time Data Collection** → Use WebSocket or HTTP endpoints with session tokens
5. **End Session** → Save all behavioral data and cleanup

### 🔄 **Returning Users (App Already Installed):**
1. **User Opens App & Enters MPIN** → **BEHAVIORAL SESSION STARTS IMMEDIATELY** → Get both auth tokens AND session tokens
2. **Real-time Data Collection** → Use WebSocket or HTTP endpoints with session tokens
3. **End Session** → Save all behavioral data and cleanup

### 🔧 **Key Implementation Details:**
- **New Users**: Use `/register` → `/login` → `/verify-mpin` flow
- **Returning Users**: Use `/mpin-login` endpoint directly (one-step authentication + session creation)
- **Token Refresh**: Use `/refresh` when access tokens expire (doesn't create new sessions)
- **Session Management**: Behavioral logging runs from MPIN entry until app close

**Perfect for Mobile Apps:** Returning users just enter MPIN and immediately start using the app with full behavioral logging!

This comprehensive API documentation provides everything your frontend needs to integrate with the behavioral logging system!
