# Session Token Database Storage Fix

## Problem Identified

The session token was being stored as `device_id` in the database instead of in the `session_token` field.

## Root Cause

In `backend/app/core/supabase_client.py`, the `create_session` method had incorrect parameter order:

### Before (Incorrect):
```python
async def create_session(self, user_id: str, session_token: str, device_info: Optional[str] = None):
    result = self.supabase.table('sessions').insert({
        'user_id': user_id,
        'session_token': session_token  # This was actually receiving device_id
    }).execute()
```

### Database Schema:
```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    device_info TEXT,        -- Should store device_id
    session_token TEXT,      -- Should store session_token
    -- ... other fields
);
```

### The Issue:
When `session_manager.create_session()` called:
```python
supabase_session = await supabase_client.create_session(
    user_id, device_id, session_token
)
```

The parameters were mapped incorrectly:
- `user_id` → `user_id` ✅
- `device_id` → `session_token` ❌ (should be `device_info`)
- `session_token` → `device_info` ❌ (should be `session_token`)

## Solution Applied

### Fixed Parameter Order:
```python
async def create_session(self, user_id: str, device_info: Optional[str] = None, session_token: Optional[str] = None):
    session_data = {
        'user_id': user_id
    }
    
    if device_info:
        session_data['device_info'] = device_info
    if session_token:
        session_data['session_token'] = session_token
        
    result = self.supabase.table('sessions').insert(session_data).execute()
```

### Now the mapping is correct:
- `user_id` → `user_id` ✅
- `device_id` → `device_info` ✅
- `session_token` → `session_token` ✅

## Files Modified

1. **`backend/app/core/supabase_client.py`**
   - Fixed `create_session` method parameter order
   - Added proper conditional field insertion
   - Made parameters optional for flexibility

## Testing

### Test File Created:
`backend/test_session_token_fix.py`

This test verifies:
1. User registration and login
2. MPIN verification (creates session with token)
3. Session status check
4. Behavioral data logging with session token
5. Proper logout

### Run Test:
```bash
cd backend
python test_session_token_fix.py
```

## Verification Steps

### 1. Check Database Records
After running the test, verify in Supabase:
```sql
SELECT id, user_id, device_info, session_token, started_at 
FROM sessions 
ORDER BY started_at DESC 
LIMIT 5;
```

### 2. Expected Results:
- `device_info` should contain device ID (e.g., "test-device-002")
- `session_token` should contain JWT token (starts with "eyJ...")
- `user_id` should contain valid UUID

### 3. Before vs After:
```
BEFORE (Incorrect):
- device_info: NULL
- session_token: "test-device-002"  ❌

AFTER (Correct):
- device_info: "test-device-002"   ✅
- session_token: "eyJhbGciOiJIUzI1NiIs..."  ✅
```

## Impact

### Fixed Issues:
1. ✅ Session tokens now stored correctly in database
2. ✅ Device information stored in proper field
3. ✅ Session token validation works correctly
4. ✅ Behavioral data logging with session tokens
5. ✅ WebSocket connections with proper token validation

### No Breaking Changes:
- All existing API endpoints continue to work
- Session management logic unchanged
- Token generation and validation unchanged

## Related Endpoints

The fix affects these endpoints that use session tokens:
- `POST /api/v1/auth/verify-mpin` - Creates session with token
- `POST /api/v1/auth/mpin-login` - Creates session with token
- `POST /api/v1/log/behavior-data` - Uses session token
- `WebSocket /api/v1/ws/behavior/{session_id}` - Uses session token
- `POST /api/v1/ws/sessions/{session_id}/lifecycle` - Uses session token

## Monitoring

### Check for Issues:
1. Monitor session creation logs
2. Verify session token validation
3. Check behavioral data logging success rate
4. Monitor WebSocket connection success

### Debug Commands:
```bash
# Check session token contents
curl http://localhost:8000/api/v1/ws/debug/token/{token}

# Check session status
curl -H "Authorization: Bearer {access_token}" \
     http://localhost:8000/api/v1/auth/session-status
```

## Conclusion

The session token database storage issue has been resolved. Session tokens are now properly stored in the `session_token` field, and device information is stored in the `device_info` field as intended.

This fix ensures proper session management, token validation, and behavioral data logging throughout the application. 