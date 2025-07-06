#!/usr/bin/env python3
"""
Test script to verify the complete authentication flow with session_id fix
"""

import sys
import os
import asyncio
sys.path.append(os.getcwd())

from app.core.security import create_session_token, extract_session_info
from app.core.session_manager import SessionManager
import uuid

async def test_complete_auth_flow():
    """Test the complete authentication flow"""
    
    print('=== Testing Complete Auth Flow ===')
    
    # Simulate the MPIN-only login flow (fixed version)
    user_id = 'test-user-123'
    phone = '1234567890'
    device_id = 'test-device-456'
    
    # Step 1: Create session manager and session
    session_manager = SessionManager()
    
    # First create session to get session_id (this is the fixed approach)
    session_id = await session_manager.create_session(user_id, phone, device_id, None)
    print(f'Step 1: Session created with ID: {session_id}')
    
    # Step 2: Create token with the session_id
    session_token = create_session_token(phone, device_id, user_id, session_id)
    print('Step 2: Session token created')
    
    # Step 3: Update session with token
    session = session_manager.get_session(session_id)
    if session:
        session.session_token = session_token
        print('Step 3: Session updated with token')
    
    # Step 4: Extract and verify session info
    session_info = extract_session_info(session_token)
    print(f'Step 4: Session info extracted: {session_info}')
    
    # Step 5: Verify everything works
    print('\n=== Verification Results ===')
    
    if session_info and session_info.get('session_id') == session_id:
        print('✅ SUCCESS: Complete auth flow working correctly!')
        print(f'   - Session ID matches: {session_info.get("session_id")}')
        print(f'   - User phone: {session_info.get("user_phone")}')
        print(f'   - User ID: {session_info.get("user_id")}')
        print(f'   - Device ID: {session_info.get("device_id")}')
        print(f'   - Created at: {session_info.get("created_at")}')
        
        # Additional verification
        if session:
            print(f'   - Session object exists: ✅')
            print(f'   - Session token stored: ✅')
            print(f'   - Session active: {"✅" if session.is_active else "❌"}')
        
        print('\n🎉 The session_id issue has been FIXED!')
        return True
    else:
        print('❌ FAILED: Auth flow has issues')
        print(f'   - Expected session_id: {session_id}')
        print(f'   - Actual session_id: {session_info.get("session_id") if session_info else "None"}')
        return False

def test_token_creation_comparison():
    """Compare token creation with and without session_id"""
    
    print('\n=== Token Creation Comparison ===')
    
    phone = '1234567890'
    device_id = 'test-device'
    user_id = 'test-user'
    session_id = str(uuid.uuid4())
    
    # Create token WITHOUT session_id (old behavior)
    token_without = create_session_token(phone, device_id, user_id)
    info_without = extract_session_info(token_without)
    
    # Create token WITH session_id (fixed behavior)
    token_with = create_session_token(phone, device_id, user_id, session_id)
    info_with = extract_session_info(token_with)
    
    print(f'Token WITHOUT session_id:')
    print(f'  - Has session_id: {"❌" if not info_without or not info_without.get("session_id") else "✅"}')
    print(f'  - Session info: {info_without}')
    
    print(f'Token WITH session_id:')
    print(f'  - Has session_id: {"✅" if info_with and info_with.get("session_id") else "❌"}')
    print(f'  - Session info: {info_with}')
    
    return info_with and info_with.get('session_id') is not None

async def main():
    """Main test function"""
    
    try:
        # Test 1: Complete auth flow
        flow_success = await test_complete_auth_flow()
        
        # Test 2: Token creation comparison
        token_success = test_token_creation_comparison()
        
        print('\n=== FINAL RESULTS ===')
        if flow_success and token_success:
            print('🎉 ALL TESTS PASSED! The session_id issue is FIXED!')
            print('✅ Extract session info now returns proper session_id')
            print('✅ Auth flow works end-to-end')
            print('✅ WebSocket authentication should now work')
        else:
            print('❌ SOME TESTS FAILED! Issue still exists.')
            
    except Exception as e:
        print(f'Error during testing: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
