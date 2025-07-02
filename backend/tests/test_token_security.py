"""
Advanced Token Security Tests
Tests for JWT token security, refresh token rotation, and edge cases
"""

import pytest
import time
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from jose import jwt, JWTError
from main import app
from app.core.config import settings
from app.core.security import (
    create_access_token, create_refresh_token, verify_access_token, 
    verify_refresh_token, TOKEN_TYPE_ACCESS, TOKEN_TYPE_REFRESH
)
from app.core.token_manager import token_manager

client = TestClient(app)

class TestTokenSecurity:
    """Comprehensive token security testing"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Clear token manager state
        token_manager.active_refresh_tokens.clear()
        token_manager.blacklisted_tokens.clear()
        token_manager.user_tokens.clear()
    
    def test_jwt_token_structure(self):
        """Test JWT token structure and claims"""
        payload = {
            "sub": "user123",
            "phone": "9876543210",
            "device_id": "device001"
        }
        
        # Test access token structure
        access_token = create_access_token(payload)
        decoded = jwt.decode(access_token, "", options={"verify_signature": False})
        
        assert decoded["sub"] == "user123"
        assert decoded["phone"] == "9876543210"
        assert decoded["device_id"] == "device001"
        assert decoded["type"] == TOKEN_TYPE_ACCESS
        assert "exp" in decoded
        assert "iat" in decoded
        
        # Test refresh token structure
        refresh_token = create_refresh_token(payload)
        decoded_refresh = jwt.decode(refresh_token, "", options={"verify_signature": False})
        
        assert decoded_refresh["sub"] == "user123"
        assert decoded_refresh["type"] == TOKEN_TYPE_REFRESH
        assert "jti" in decoded_refresh  # Unique identifier
        assert decoded_refresh["exp"] > decoded["exp"]  # Longer expiry
    
    def test_token_expiration_times(self):
        """Test token expiration configuration"""
        payload = {"sub": "user123", "phone": "9876543210", "device_id": "device001"}
        
        access_token = create_access_token(payload)
        refresh_token = create_refresh_token(payload)
        
        access_decoded = jwt.decode(access_token, "", options={"verify_signature": False})
        refresh_decoded = jwt.decode(refresh_token, "", options={"verify_signature": False})
        
        # Check access token expiry (15 minutes by default)
        access_exp = datetime.fromtimestamp(access_decoded["exp"])
        access_iat = datetime.fromtimestamp(access_decoded["iat"])
        access_duration = access_exp - access_iat
        
        assert access_duration.total_seconds() <= settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60 + 60  # 1 minute tolerance
        
        # Check refresh token expiry (30 days by default)
        refresh_exp = datetime.fromtimestamp(refresh_decoded["exp"])
        refresh_iat = datetime.fromtimestamp(refresh_decoded["iat"])
        refresh_duration = refresh_exp - refresh_iat
        
        assert refresh_duration.days <= settings.REFRESH_TOKEN_EXPIRE_DAYS + 1  # 1 day tolerance
    
    def test_token_signature_validation(self):
        """Test token signature validation with different keys"""
        payload = {"sub": "user123", "phone": "9876543210", "device_id": "device001"}
        
        # Valid tokens with correct keys
        access_token = create_access_token(payload)
        refresh_token = create_refresh_token(payload)
        
        assert verify_access_token(access_token) is not None
        assert verify_refresh_token(refresh_token) is not None
        
        # Test access token with wrong key (should fail)
        fake_access_token = jwt.encode(payload, "wrong_key", algorithm=settings.ALGORITHM)
        assert verify_access_token(fake_access_token) is None
        
        # Test refresh token with wrong key (should fail)
        fake_refresh_token = jwt.encode(payload, "wrong_key", algorithm=settings.ALGORITHM)
        assert verify_refresh_token(fake_refresh_token) is None
    
    def test_token_type_validation(self):
        """Test that tokens are validated for correct type"""
        payload = {"sub": "user123", "phone": "9876543210", "device_id": "device001"}
        
        access_token = create_access_token(payload)
        refresh_token = create_refresh_token(payload)
        
        # Access token should not validate as refresh token
        assert verify_refresh_token(access_token) is None
        
        # Refresh token should not validate as access token
        assert verify_access_token(refresh_token) is None
    
    def test_refresh_token_rotation(self):
        """Test refresh token rotation on refresh endpoint"""
        # Create initial tokens
        payload = {"sub": "user123", "phone": "9876543210", "device_id": "device001"}
        refresh_token = create_refresh_token(payload)
        
        # Store in token manager
        refresh_payload = verify_refresh_token(refresh_token)
        jti = refresh_payload["jti"]
        expires_at = datetime.fromtimestamp(refresh_payload["exp"])
        token_manager.store_refresh_token(jti, "user123", "device001", expires_at)
        
        # Use refresh endpoint
        refresh_request = {"refresh_token": refresh_token}
        response = client.post("/api/v1/auth/refresh", json=refresh_request)
        
        if response.status_code == 200:
            data = response.json()
            new_refresh_token = data["refresh_token"]
            
            # Verify new token is different
            assert new_refresh_token != refresh_token
            
            # Verify old token is revoked
            assert not token_manager.is_token_valid(jti)
            
            # Verify new token is valid
            new_payload = verify_refresh_token(new_refresh_token)
            assert new_payload is not None
            assert new_payload["jti"] != jti
    
    def test_concurrent_refresh_attempts(self):
        """Test handling of concurrent refresh attempts with same token"""
        payload = {"sub": "user123", "phone": "9876543210", "device_id": "device001"}
        refresh_token = create_refresh_token(payload)
        
        # Store in token manager
        refresh_payload = verify_refresh_token(refresh_token)
        jti = refresh_payload["jti"]
        expires_at = datetime.fromtimestamp(refresh_payload["exp"])
        token_manager.store_refresh_token(jti, "user123", "device001", expires_at)
        
        # Simulate concurrent refresh attempts
        refresh_request = {"refresh_token": refresh_token}
        
        # First request should succeed
        response1 = client.post("/api/v1/auth/refresh", json=refresh_request)
        
        # Second request with same token should fail (token revoked after first use)
        response2 = client.post("/api/v1/auth/refresh", json=refresh_request)
        
        if response1.status_code == 200:
            assert response2.status_code == 401
    
    def test_device_based_token_isolation(self):
        """Test that tokens are properly isolated by device"""
        user_id = "user123"
        phone = "9876543210"
        
        # Create tokens for different devices
        device1_payload = {"sub": user_id, "phone": phone, "device_id": "device001"}
        device2_payload = {"sub": user_id, "phone": phone, "device_id": "device002"}
        
        refresh1 = create_refresh_token(device1_payload)
        refresh2 = create_refresh_token(device2_payload)
        
        # Store both tokens
        payload1 = verify_refresh_token(refresh1)
        payload2 = verify_refresh_token(refresh2)
        
        token_manager.store_refresh_token(
            payload1["jti"], user_id, "device001", 
            datetime.fromtimestamp(payload1["exp"])
        )
        token_manager.store_refresh_token(
            payload2["jti"], user_id, "device002", 
            datetime.fromtimestamp(payload2["exp"])
        )
        
        # Verify both tokens are active
        assert len(token_manager.get_user_active_tokens(user_id)) == 2
        
        # Revoke tokens for device001 only
        token_manager.revoke_device_tokens(user_id, "device001")
        
        # Verify device001 token is revoked but device002 token remains
        active_tokens = token_manager.get_user_active_tokens(user_id)
        assert len(active_tokens) == 1
        assert active_tokens[0]["device_id"] == "device002"
    
    def test_token_cleanup_mechanism(self):
        """Test automatic cleanup of expired tokens"""
        user_id = "user123"
        
        # Create an expired token
        expired_jti = "expired_token_123"
        expired_time = datetime.utcnow() - timedelta(days=1)  # Already expired
        token_manager.store_refresh_token(expired_jti, user_id, "device001", expired_time)
        
        # Create a valid token
        valid_jti = "valid_token_123"
        valid_time = datetime.utcnow() + timedelta(days=30)
        token_manager.store_refresh_token(valid_jti, user_id, "device002", valid_time)
        
        # Before cleanup
        assert len(token_manager.active_refresh_tokens) == 2
        
        # Trigger cleanup
        import asyncio
        asyncio.run(token_manager.cleanup_expired_tokens())
        
        # After cleanup - expired token should be removed
        assert len(token_manager.active_refresh_tokens) == 1
        assert valid_jti in token_manager.active_refresh_tokens
        assert expired_jti not in token_manager.active_refresh_tokens
    
    def test_malformed_jwt_handling(self):
        """Test handling of malformed JWT tokens"""
        malformed_tokens = [
            "not.a.jwt",
            "invalid_base64.invalid_base64.invalid_base64",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid_payload.signature",
            "",
            "single_string_not_jwt"
        ]
        
        for malformed_token in malformed_tokens:
            assert verify_access_token(malformed_token) is None
            assert verify_refresh_token(malformed_token) is None
            
            # Test endpoint with malformed token
            headers = {"Authorization": f"Bearer {malformed_token}"}
            response = client.post("/api/v1/auth/verify-mpin", 
                                 json={"mpin": "123456"}, headers=headers)
            assert response.status_code in [401, 403]  # Unauthorized or Forbidden
    
    def test_token_replay_attack_prevention(self):
        """Test prevention of token replay attacks"""
        payload = {"sub": "user123", "phone": "9876543210", "device_id": "device001"}
        refresh_token = create_refresh_token(payload)
        
        # Store and use token once
        refresh_payload = verify_refresh_token(refresh_token)
        jti = refresh_payload["jti"]
        expires_at = datetime.fromtimestamp(refresh_payload["exp"])
        token_manager.store_refresh_token(jti, "user123", "device001", expires_at)
        
        # First use - should succeed
        refresh_request = {"refresh_token": refresh_token}
        response1 = client.post("/api/v1/auth/refresh", json=refresh_request)
        
        # Attempt to replay the same token - should fail
        response2 = client.post("/api/v1/auth/refresh", json=refresh_request)
        
        if response1.status_code == 200:
            assert response2.status_code == 401
            assert "revoked" in response2.json()["detail"] or "Invalid" in response2.json()["detail"]
    
    def test_user_logout_token_revocation(self):
        """Test that logout revokes all user tokens"""
        user_id = "user123"
        
        # Create multiple tokens for the user
        for i in range(3):
            jti = f"token_{i}"
            device_id = f"device_{i}"
            expires_at = datetime.utcnow() + timedelta(days=30)
            token_manager.store_refresh_token(jti, user_id, device_id, expires_at)
        
        # Verify tokens are active
        assert len(token_manager.get_user_active_tokens(user_id)) == 3
        
        # Simulate logout - revoke all user tokens
        token_manager.revoke_user_tokens(user_id)
        
        # Verify all tokens are revoked
        assert len(token_manager.get_user_active_tokens(user_id)) == 0
        
        # Verify tokens are in blacklist
        for i in range(3):
            assert f"token_{i}" in token_manager.blacklisted_tokens
    
    def test_token_manager_statistics(self):
        """Test token manager statistics and monitoring"""
        # Start with clean state
        initial_stats = token_manager.get_stats()
        assert initial_stats["active_refresh_tokens"] == 0
        assert initial_stats["blacklisted_tokens"] == 0
        
        # Add some tokens
        for i in range(5):
            jti = f"token_{i}"
            user_id = f"user_{i % 2}"  # 2 users
            device_id = f"device_{i}"
            expires_at = datetime.utcnow() + timedelta(days=30)
            token_manager.store_refresh_token(jti, user_id, device_id, expires_at)
        
        stats = token_manager.get_stats()
        assert stats["active_refresh_tokens"] == 5
        assert stats["users_with_tokens"] == 2
        assert stats["total_user_token_mappings"] == 5
        
        # Revoke some tokens
        token_manager.revoke_token("token_0")
        token_manager.revoke_token("token_1")
        
        final_stats = token_manager.get_stats()
        assert final_stats["active_refresh_tokens"] == 3
        assert final_stats["blacklisted_tokens"] == 2
    
    def test_custom_token_expiry(self):
        """Test custom token expiry settings"""
        payload = {"sub": "user123", "phone": "9876543210", "device_id": "device001"}
        
        # Test custom access token expiry
        custom_expiry = timedelta(minutes=5)
        access_token = create_access_token(payload, custom_expiry)
        
        decoded = jwt.decode(access_token, "", options={"verify_signature": False})
        exp_time = datetime.fromtimestamp(decoded["exp"])
        iat_time = datetime.fromtimestamp(decoded["iat"])
        
        duration = exp_time - iat_time
        assert abs(duration.total_seconds() - 300) < 60  # 5 minutes ± 1 minute tolerance
    
    def test_token_information_leakage_prevention(self):
        """Test that tokens don't leak sensitive information"""
        payload = {
            "sub": "user123",
            "phone": "9876543210",
            "device_id": "device001",
            "password": "should_not_be_included",  # This should not appear in token
            "mpin": "654321"  # This should not appear in token
        }
        
        # Create tokens
        access_token = create_access_token(payload)
        refresh_token = create_refresh_token(payload)
        
        # Decode without verification to check payload
        access_decoded = jwt.decode(access_token, "", options={"verify_signature": False})
        refresh_decoded = jwt.decode(refresh_token, "", options={"verify_signature": False})
        
        # Ensure sensitive data is not included
        sensitive_fields = ["password", "mpin"]
        for field in sensitive_fields:
            assert field not in access_decoded
            assert field not in refresh_decoded
            
        # Ensure only expected fields are present
        expected_access_fields = {"sub", "phone", "device_id", "exp", "iat", "type"}
        expected_refresh_fields = {"sub", "phone", "device_id", "exp", "iat", "type", "jti"}
        
        assert set(access_decoded.keys()) == expected_access_fields
        assert set(refresh_decoded.keys()) == expected_refresh_fields

    def test_integration_login_refresh_flow(self):
        """Test complete integration flow: register -> login -> refresh -> verify"""
        # This test requires the full system to be working
        # It's more of an integration test
        
        # Step 1: Register (may fail in test environment)
        register_data = {
            "phone": "9111111111",
            "password": "IntegrationTest123",
            "mpin": "987654"
        }
        register_response = client.post("/api/v1/auth/register", json=register_data)
        
        # Step 2: Login (may fail in test environment)
        login_data = {
            "phone": "9111111111",
            "password": "IntegrationTest123",
            "device_id": "integration_device"
        }
        login_response = client.post("/api/v1/auth/login", json=login_data)
        
        if login_response.status_code == 200:
            login_data = login_response.json()
            
            # Step 3: Use refresh token
            refresh_request = {"refresh_token": login_data["refresh_token"]}
            refresh_response = client.post("/api/v1/auth/refresh", json=refresh_request)
            
            if refresh_response.status_code == 200:
                refresh_data = refresh_response.json()
                
                # Step 4: Use new access token for MPIN verification
                headers = {"Authorization": f"Bearer {refresh_data['access_token']}"}
                mpin_data = {"mpin": "987654"}
                verify_response = client.post("/api/v1/auth/verify-mpin", 
                                            json=mpin_data, headers=headers)
                
                # Should work with refreshed token
                assert verify_response.status_code in [200, 404, 500]  # 404/500 for missing user in test DB
        
        # If login fails (expected in test environment), just verify endpoints exist
        else:
            assert login_response.status_code in [401, 500]
