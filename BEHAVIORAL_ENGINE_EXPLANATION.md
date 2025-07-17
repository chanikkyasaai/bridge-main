# 🧠 How Your Behavioral Authentication Engine Works

## 📊 System Architecture Overview

Your behavioral authentication engine is a sophisticated multi-layered security system that continuously monitors user behavior to detect fraud and authenticate users in real-time. Here's exactly how it works:

## 🔄 Complete Data Flow

### 1. **User Login & Session Creation**
```
User enters MPIN → Backend validates → Creates session → Starts ML tracking
                                    ↓
              Session ID + Token generated → WebSocket connection established
```

### 2. **Real-Time Behavioral Streaming**
```
User interacts with app → Behavioral events collected → Sent via WebSocket
                                                      ↓
                        Events processed by backend → Analyzed by ML Engine
```

### 3. **Risk Assessment & Action**
```
ML analysis + Rule-based scoring → Risk score calculated → Security action taken
                                                        ↓
                              Normal (0-0.6) → Continue session
                              Suspicious (0.7-0.8) → Request MPIN
                              High Risk (0.9+) → Block session
```

## 🎯 Core Components Explained

### **1. WebSocket Behavioral Streaming** (`websocket.py`)

**Purpose**: Real-time collection of user behavioral data

**How it works**:
- User connects to `/api/v1/behavior/{session_id}?token={session_token}`
- Frontend streams behavioral events (typing, touch, navigation)
- Backend processes each event and updates risk score
- Triggers security actions based on risk level

**Key Events Monitored**:
- **Typing patterns**: Speed, rhythm, keystroke intervals
- **Touch behavior**: Pressure, duration, coordinates
- **Navigation patterns**: Page flow, scroll behavior
- **Transaction context**: Amount, timing, beneficiary

### **2. Session Risk Management** (`session_manager.py`)

**Purpose**: Manages user sessions and tracks risk scores

**Risk Thresholds**:
- **Normal (0.0-0.6)**: Continue monitoring
- **Suspicious (0.7-0.8)**: Request MPIN re-authentication
- **High Risk (0.9-1.0)**: Block session immediately

**Security Actions**:
```python
def update_risk_score(self, new_score: float):
    if new_score >= 0.9:           # HIGH_RISK_THRESHOLD
        self.block_session()       # Force logout
    elif new_score >= 0.7:         # SUSPICIOUS_THRESHOLD  
        self.request_mpin_verification()  # Show MPIN dialog
```

### **3. ML Engine Integration** (`ml_hooks.py`)

**Purpose**: Connects backend to ML analysis engine

**Process**:
1. Behavioral events sent to ML Engine (port 8001)
2. ML Engine returns decision: `allow`, `challenge`, or `block`
3. Backend combines ML decision with rule-based scoring
4. Final risk score determines security action

**ML Decisions**:
- **High confidence (>0.8)**: Direct action taken
- **Medium confidence (0.6-0.8)**: Risk adjustment applied
- **Low confidence (<0.6)**: Fall back to rule-based scoring

### **4. ML Engine Analysis** (`ml_engine_api_service.py`)

**Purpose**: Advanced behavioral pattern analysis

**Two-Phase System**:

**Phase 1 - Learning System**:
- Collects user behavioral patterns
- Builds user-specific behavioral profile
- Uses FAISS (Facebook AI Similarity Search) for pattern matching

**Phase 2 - Continuous Analysis**:
- Real-time comparison against learned patterns
- Adaptive learning from new behavioral data
- Anomaly detection for unusual patterns

**Analysis Process**:
```python
# For each behavioral event:
1. Extract behavioral features
2. Compare against user's historical patterns
3. Calculate similarity score
4. Determine risk level based on deviation
5. Return decision with confidence score
```

## 🛡️ Multi-Layer Security Analysis

### **Layer 1: Rule-Based Risk Scoring**

Immediate risk factors with predefined weights:
```python
risk_factors = {
    "rapid_clicks": +0.1,           # Automated behavior
    "large_transaction": +0.2,       # High-value transfer
    "new_beneficiary": +0.15,        # Unknown recipient
    "off_hours_activity": +0.1,      # Unusual timing
    "mpin_failed": +0.25,           # Failed authentication
    "mpin_verified": -0.1,          # Successful verification
    "normal_typing": -0.02,         # Expected behavior
}
```

### **Layer 2: ML Pattern Analysis**

Advanced behavioral modeling:
- **FAISS Vector Similarity**: Compares current behavior to learned patterns
- **Adaptive Learning**: Continuously updates user behavioral model
- **Anomaly Detection**: Identifies significant deviations from normal behavior

### **Layer 3: Context-Aware Analysis**

Transaction and environmental factors:
- **Transaction amount**: Higher amounts increase risk
- **Time of day**: Off-hours activity flagged
- **Device behavior**: Screen orientation, device movement
- **Session patterns**: Navigation flow, dwell times

## 🔄 Real-Time Processing Flow

### **Step 1: Event Collection**
```javascript
// Frontend sends behavioral event
{
    "event_type": "typing_pattern",
    "features": {
        "typing_speed": 65,
        "keystroke_intervals": [110, 120, 105],
        "accuracy": 0.95
    }
}
```

### **Step 2: Backend Processing**
```python
# WebSocket receives event
behavioral_event = json.loads(data)

# Store in session buffer
session.add_behavioral_data(event_type, event_data)

# Send to ML Engine for analysis
ml_response = await behavioral_event_hook(user_id, session_id, recent_events)

# Update risk score
await analyze_behavioral_pattern(session, event_type, event_data, ml_response)
```

### **Step 3: ML Engine Analysis**
```python
# ML Engine processes behavioral pattern
features = extract_behavioral_features(events)
similarity_score = compare_with_user_profile(user_id, features)
decision = determine_security_action(similarity_score)

return {
    "decision": "allow|challenge|block",
    "confidence": 0.85,
    "similarity_score": 0.92
}
```

### **Step 4: Security Action**
```python
# Backend takes action based on risk score
if risk_score >= 0.9:
    # Send WebSocket message to frontend
    await websocket.send({
        "type": "session_blocked",
        "reason": "High risk behavior detected"
    })
    # Frontend: Force logout, show security message

elif risk_score >= 0.7:
    # Send WebSocket message to frontend  
    await websocket.send({
        "type": "mpin_required",
        "message": "Please verify your MPIN"
    })
    # Frontend: Show MPIN dialog
```

## 📊 Behavioral Data Storage

### **Session-Level Storage**:
- **In-Memory Buffer**: Real-time behavioral events during session
- **Risk Score Tracking**: Continuous risk updates
- **Event History**: Complete behavioral timeline

### **Permanent Storage**:
- **Supabase Database**: Session summaries and risk scores
- **Behavioral Logs**: Detailed event history for analysis
- **User Profiles**: Long-term behavioral patterns

## 🎯 Key Features

### **✅ Real-Time Processing**
- Events processed as they occur
- Immediate risk assessment
- Instant security responses

### **✅ Adaptive Learning**  
- User behavior models improve over time
- Reduced false positives as system learns
- Personalized risk thresholds

### **✅ Multi-Modal Analysis**
- Typing patterns
- Touch behavior  
- Navigation habits
- Transaction context
- Device characteristics

### **✅ Graduated Response**
- Normal behavior: Continue monitoring
- Suspicious behavior: Request additional authentication
- High-risk behavior: Block session immediately

## 🚀 Production Benefits

1. **Fraud Prevention**: Detects account takeovers and automated attacks
2. **User Experience**: Seamless security without friction for legitimate users
3. **Adaptive Security**: Continuously improves accuracy over time
4. **Compliance**: Provides detailed audit trail of security decisions
5. **Scalability**: Handles real-time analysis for thousands of users

## 🔧 Technical Implementation

Your system is fully integrated and operational with:
- **Backend API**: FastAPI running on port 8000
- **ML Engine**: Independent service on port 8001  
- **WebSocket**: Real-time behavioral streaming
- **Database**: Supabase for persistence
- **Risk Thresholds**: Configurable security levels

The entire system works together to provide **continuous, invisible authentication** that protects users while maintaining a smooth banking experience! 🛡️✨
