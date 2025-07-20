# Device Context Integration

## Overview

The backend now supports enhanced device context information during MPIN verification and login. This allows the frontend to send comprehensive device information for better security analysis and behavioral tracking.

## Request Format

### MPIN Verification with Device Context

```json
{
  "mpin": "12345",
  "context": {
    "device_id": "device-123",
    "device_type": "mobile",
    "device_model": "iPhone 15 Pro",
    "os_version": "iOS 17.2",
    "app_version": "1.2.3",
    "network_type": "wifi",
    "location_data": {
      "latitude": 12.9716,
      "longitude": 77.5946,
      "city": "Bangalore",
      "country": "India"
    },
    "user_agent": "CanaraBankApp/1.2.3 (iPhone; iOS 17.2; Scale/3.00)",
    "ip_address": "192.168.1.100"
  }
}
```

### MPIN Login with Device Context

```json
{
  "phone": "9876543210",
  "mpin": "12345",
  "device_id": "device-123",
  "context": {
    "device_id": "device-123",
    "device_type": "tablet",
    "device_model": "iPad Pro 12.9",
    "os_version": "iPadOS 17.2",
    "app_version": "1.2.3",
    "network_type": "cellular",
    "location_data": {
      "latitude": 19.0760,
      "longitude": 72.8777,
      "city": "Mumbai",
      "country": "India"
    },
    "user_agent": "CanaraBankApp/1.2.3 (iPad; iPadOS 17.2; Scale/2.00)",
    "ip_address": "10.0.0.50"
  }
}
```

## Device Context Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `device_id` | string | ✅ | Unique device identifier |
| `device_type` | string | ❌ | Device type (mobile, tablet, desktop) |
| `device_model` | string | ❌ | Device model name |
| `os_version` | string | ❌ | Operating system version |
| `app_version` | string | ❌ | Application version |
| `network_type` | string | ❌ | Network connection type (wifi, cellular, ethernet) |
| `location_data` | object | ❌ | GPS location information |
| `user_agent` | string | ❌ | Browser/App user agent string |
| `ip_address` | string | ❌ | Device IP address |

## Backend Processing

### 1. Device Context Logging

When device context is provided, it's logged as behavioral data:

```python
session.add_behavioral_data("device_context", {
    "session_id": session_id,
    "device_id": device_context.device_id,
    "device_type": device_context.device_type,
    "device_model": device_context.device_model,
    "os_version": device_context.os_version,
    "app_version": device_context.app_version,
    "network_type": device_context.network_type,
    "location_data": device_context.location_data,
    "user_agent": device_context.user_agent,
    "ip_address": device_context.ip_address,
    "timestamp": session.last_activity.isoformat()
})
```

### 2. ML Engine Integration

Enhanced device information is sent to the ML Engine:

```python
ml_device_info = {
    "device_id": device_id,
    "phone": phone,
    "session_id": session_id,
    "device_type": device_context.device_type,
    "device_model": device_context.device_model,
    "os_version": device_context.os_version,
    "app_version": device_context.app_version,
    "network_type": device_context.network_type,
    "ip_address": device_context.ip_address
}
```

### 3. Fallback Behavior

If no device context is provided, the system falls back to:
- Using `device_id` from the access token
- Minimal device information for ML Engine
- Standard session creation

## API Endpoints Updated

### 1. `POST /api/v1/auth/verify-mpin`

**Request:**
```json
{
  "mpin": "12345",
  "context": {
    "device_id": "device-123",
    "device_type": "mobile",
    "device_model": "iPhone 15 Pro",
    "os_version": "iOS 17.2",
    "app_version": "1.2.3",
    "network_type": "wifi",
    "location_data": {
      "latitude": 12.9716,
      "longitude": 77.5946,
      "city": "Bangalore",
      "country": "India"
    },
    "user_agent": "CanaraBankApp/1.2.3 (iPhone; iOS 17.2; Scale/3.00)",
    "ip_address": "192.168.1.100"
  }
}
```

**Response:**
```json
{
  "message": "MPIN verified successfully",
  "user_id": "uuid",
  "phone": "9876543210",
  "status": "verified",
  "session_id": "session-uuid",
  "session_token": "jwt-token",
  "behavioral_logging": "started"
}
```

### 2. `POST /api/v1/auth/mpin-login`

**Request:**
```json
{
  "phone": "9876543210",
  "mpin": "12345",
  "device_id": "device-123",
  "context": {
    "device_id": "device-123",
    "device_type": "tablet",
    "device_model": "iPad Pro 12.9",
    "os_version": "iPadOS 17.2",
    "app_version": "1.2.3",
    "network_type": "cellular",
    "location_data": {
      "latitude": 19.0760,
      "longitude": 72.8777,
      "city": "Mumbai",
      "country": "India"
    },
    "user_agent": "CanaraBankApp/1.2.3 (iPad; iPadOS 17.2; Scale/2.00)",
    "ip_address": "10.0.0.50"
  }
}
```

**Response:**
```json
{
  "access_token": "jwt-access-token",
  "refresh_token": "jwt-refresh-token",
  "token_type": "bearer",
  "expires_in": 900,
  "session_id": "session-uuid",
  "session_token": "jwt-session-token",
  "behavioral_logging": "started",
  "message": "MPIN login successful - behavioral logging started"
}
```

## Security Benefits

### 1. Enhanced Risk Assessment
- Device fingerprinting for anomaly detection
- Location-based security analysis
- Network type monitoring
- App version tracking

### 2. Behavioral Analysis
- Device-specific behavioral patterns
- OS-specific interaction patterns
- Network-dependent behavior analysis

### 3. Fraud Detection
- Suspicious device changes
- Unusual location patterns
- Network type anomalies
- Device model inconsistencies

## Testing

### Test File: `backend/test_device_context.py`

This test covers:
1. **Full Context**: Complete device information
2. **Minimal Context**: Only required fields
3. **No Context**: Fallback behavior
4. **MPIN Login**: Context in MPIN-only login
5. **Behavioral Data**: Session token validation

### Run Test:
```bash
cd backend
python test_device_context.py
```

## Frontend Integration

### JavaScript Example

```javascript
// Collect device context
const deviceContext = {
  device_id: deviceId,
  device_type: deviceType,
  device_model: deviceModel,
  os_version: osVersion,
  app_version: appVersion,
  network_type: networkType,
  location_data: locationData,
  user_agent: userAgent,
  ip_address: deviceIp,
};

// MPIN verification with context
const response = await fetch('/api/v1/auth/verify-mpin', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${accessToken}`
  },
  body: JSON.stringify({
    mpin: mpin,
    context: deviceContext
  })
});
```

### React Native Example

```javascript
import DeviceInfo from 'react-native-device-info';
import Geolocation from '@react-native-community/geolocation';

const getDeviceContext = async () => {
  const location = await new Promise((resolve, reject) => {
    Geolocation.getCurrentPosition(
      position => resolve(position.coords),
      error => reject(error),
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 10000 }
    );
  });

  return {
    device_id: await DeviceInfo.getUniqueId(),
    device_type: DeviceInfo.getDeviceType(),
    device_model: DeviceInfo.getModel(),
    os_version: DeviceInfo.getSystemVersion(),
    app_version: DeviceInfo.getVersion(),
    network_type: await DeviceInfo.getCarrier(),
    location_data: {
      latitude: location.latitude,
      longitude: location.longitude
    },
    user_agent: DeviceInfo.getUserAgent(),
    ip_address: await DeviceInfo.getIpAddress()
  };
};

const verifyMPIN = async (mpin) => {
  const context = await getDeviceContext();
  
  const response = await fetch('/api/v1/auth/verify-mpin', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${accessToken}`
    },
    body: JSON.stringify({
      mpin: mpin,
      context: context
    })
  });
  
  return response.json();
};
```

## Monitoring

### Behavioral Events to Monitor
- `device_context` - Device context logged
- `ml_session_started` - ML session with enhanced device info
- `mpin_verified` - MPIN verification with context

### Database Queries

```sql
-- Check device context usage
SELECT 
  session_id,
  event_type,
  data->>'device_type' as device_type,
  data->>'device_model' as device_model,
  data->>'os_version' as os_version,
  data->>'network_type' as network_type,
  created_at
FROM behavioral_events 
WHERE event_type = 'device_context'
ORDER BY created_at DESC;

-- Analyze device patterns
SELECT 
  data->>'device_type' as device_type,
  data->>'network_type' as network_type,
  COUNT(*) as usage_count
FROM behavioral_events 
WHERE event_type = 'device_context'
GROUP BY data->>'device_type', data->>'network_type';
```

## Best Practices

### 1. Privacy Compliance
- Only collect necessary device information
- Ensure user consent for location data
- Anonymize sensitive information

### 2. Data Validation
- Validate device information on frontend
- Sanitize user agent strings
- Verify location data accuracy

### 3. Security Considerations
- Encrypt sensitive device information
- Implement rate limiting for context updates
- Monitor for suspicious device changes

### 4. Performance
- Cache device context when appropriate
- Minimize context update frequency
- Optimize data transmission size

## Conclusion

The device context integration enhances security analysis by providing comprehensive device information during authentication. This enables better fraud detection, behavioral analysis, and risk assessment while maintaining backward compatibility with existing implementations. 