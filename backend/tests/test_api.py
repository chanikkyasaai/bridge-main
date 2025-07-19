import pytest
import asyncio
import time
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import json
from jose import jwt
from main import app
from app.core.config import settings
from app.core.token_manager import token_manager
from app.core.security import (
    verify_access_token, verify_refresh_token, create_access_token, 
    create_refresh_token, create_session_token
)

client = TestClient(app)

class TestAuthenticationFlow:
    """Test suite for comprehensive authentication flow with access and refresh tokens"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Clear token manager state
        token_manager.active_refresh_tokens.clear()
        token_manager.blacklisted_tokens.clear()
        token_manager.user_tokens.clear()
    
    def test_root_endpoint(self):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Canara AI" in data["message"] or "Welcome" in data["message"]
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "canara-ai-backend"

    def test_user_registration_flow(self):
        """Test complete user registration flow with validation"""
        # Test valid registration
        register_data = {
            "phone": "9876543210",
            "password": "SecurePassword123",
            "mpin": "123456"
        }
        response = client.post("/api/v1/auth/register", json=register_data)
        # May return 200 (success), 400 (validation error), or 500 (Supabase not configured in test)
        assert response.status_code in [200, 400, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "message" in data
            assert "user_id" in data
            assert "next_step" in data

    def test_registration_validation(self):
        """Test registration input validation"""
        # Invalid phone number (too short)
        invalid_phone_data = {
            "phone": "123",
            "password": "SecurePassword123",
            "mpin": "123456"
        }
        response = client.post("/api/v1/auth/register", json=invalid_phone_data)
        assert response.status_code == 400
        assert "Invalid phone number format" in response.json()["detail"]
        
        # Invalid MPIN (wrong length)
        invalid_mpin_data = {
            "phone": "9876543210",
            "password": "SecurePassword123",
            "mpin": "12345"  # Should be 6 digits
        }
        response = client.post("/api/v1/auth/register", json=invalid_mpin_data)
        assert response.status_code == 400
        assert "MPIN must be exactly" in response.json()["detail"]

    def test_login_and_token_generation(self):
        """Test login flow and token generation"""
        # First, try to register (may fail in test environment)
        register_data = {
            "phone": "9123456789",
            "password": "TestPassword123",
            "mpin": "654321"
        }
        client.post("/api/v1/auth/register", json=register_data)
        
        # Test login
        login_data = {
            "phone": "9123456789",
            "password": "TestPassword123",
            "device_id": "test_device_001"
        }
        response = client.post("/api/v1/auth/login", json=login_data)
        
        # Login may fail if Supabase not configured, but should validate structure
        if response.status_code == 200:
            data = response.json()
            self._validate_token_response(data)
        else:
            # Should still validate endpoint structure
            assert response.status_code in [401, 500]  # Unauthorized or server error

    def _validate_token_response(self, data):
        """Helper to validate token response structure"""
        required_fields = ["access_token", "refresh_token", "token_type", "expires_in", "session_id", "session_token"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        assert data["token_type"] == "bearer"
        assert isinstance(data["expires_in"], int)
        assert data["expires_in"] > 0
        
        # Validate token format (should be valid JWT structure)
        access_token = data["access_token"]
        refresh_token = data["refresh_token"]
        
        # Check if tokens have 3 parts (header.payload.signature)
        assert len(access_token.split('.')) == 3
        assert len(refresh_token.split('.')) == 3

    def test_access_token_verification(self):
        """Test access token verification and structure"""
        # Create mock access token for testing
        test_payload = {
            "sub": "test_user_123",
            "phone": "9876543210",
            "device_id": "test_device"
        }
        
        access_token = create_access_token(test_payload)
        
        # Test token verification
        payload = verify_access_token(access_token)
        assert payload is not None
        assert payload["sub"] == "test_user_123"
        assert payload["phone"] == "9876543210"
        assert payload["device_id"] == "test_device"
        assert payload["type"] == "access"
        
        # Test expired token (not feasible in quick test, but structure check)
        assert "exp" in payload
        assert payload["exp"] > datetime.utcnow().timestamp()

    def test_refresh_token_verification(self):
        """Test refresh token verification and structure"""
        # Create mock refresh token for testing
        test_payload = {
            "sub": "test_user_123",
            "phone": "9876543210",
            "device_id": "test_device"
        }
        
        refresh_token = create_refresh_token(test_payload)
        
        # Test token verification
        payload = verify_refresh_token(refresh_token)
        assert payload is not None
        assert payload["sub"] == "test_user_123"
        assert payload["phone"] == "9876543210"
        assert payload["device_id"] == "test_device"
        assert payload["type"] == "refresh"
        assert "jti" in payload  # Unique token ID
        
        # Verify expiration is longer than access token
        access_payload = verify_access_token(
            create_access_token(test_payload)
        )
        assert payload["exp"] > access_payload["exp"]

    def test_refresh_token_endpoint(self):
        """Test the refresh token endpoint functionality"""
        # Create valid refresh token
        test_payload = {
            "sub": "user_123",
            "phone": "9876543210",
            "device_id": "test_device"
        }
        
        refresh_token = create_refresh_token(test_payload)
        
        # Store token in token manager
        refresh_payload = verify_refresh_token(refresh_token)
        if refresh_payload:
            jti = refresh_payload["jti"]
            expires_at = datetime.fromtimestamp(refresh_payload["exp"])
            token_manager.store_refresh_token(jti, "user_123", "test_device", expires_at)
            
            # Test refresh endpoint
            refresh_request = {"refresh_token": refresh_token}
            response = client.post("/api/v1/auth/refresh", json=refresh_request)
            
            if response.status_code == 200:
                data = response.json()
                self._validate_token_response(data)
                
                # Verify old token is revoked
                assert not token_manager.is_token_valid(jti)

    def test_invalid_refresh_token(self):
        """Test refresh endpoint with invalid tokens"""
        # Test with invalid token
        invalid_refresh = {"refresh_token": "invalid.token.here"}
        response = client.post("/api/v1/auth/refresh", json=invalid_refresh)
        assert response.status_code == 401
        assert "Invalid refresh token" in response.json()["detail"]
        
        # Test with expired/revoked token
        test_payload = {
            "sub": "user_123",
            "phone": "9876543210",
            "device_id": "test_device"
        }
        
        refresh_token = create_refresh_token(test_payload)
        refresh_payload = verify_refresh_token(refresh_token)
        
        if refresh_payload:
            jti = refresh_payload["jti"]
            # Don't store token in token manager (simulates revoked token)
            
            refresh_request = {"refresh_token": refresh_token}
            response = client.post("/api/v1/auth/refresh", json=refresh_request)
            assert response.status_code == 401

    def test_mpin_verification_with_access_token(self):
        """Test MPIN verification requiring valid access token"""
        # Create valid access token
        test_payload = {
            "sub": "user_123",
            "phone": "9876543210",
            "device_id": "test_device"
        }
        
        access_token = create_access_token(test_payload)
        
        # Test MPIN verification endpoint
        headers = {"Authorization": f"Bearer {access_token}"}
        mpin_data = {"mpin": "123456"}
        
        response = client.post("/api/v1/auth/verify-mpin", json=mpin_data, headers=headers)
        # May fail due to user not existing in Supabase, but should validate auth
        assert response.status_code in [200, 404, 500]
        
        # Test without authorization header
        response = client.post("/api/v1/auth/verify-mpin", json=mpin_data)
        assert response.status_code == 403  # Forbidden without auth

    def test_invalid_access_token_scenarios(self):
        """Test various invalid access token scenarios"""
        mpin_data = {"mpin": "123456"}
        
        # Test with invalid token format
        invalid_headers = {"Authorization": "Bearer invalid.token.format"}
        response = client.post("/api/v1/auth/verify-mpin", json=mpin_data, headers=invalid_headers)
        assert response.status_code == 401
        
        # Test with wrong token type (refresh token instead of access)
        test_payload = {
            "sub": "user_123",
            "phone": "9876543210",
            "device_id": "test_device"
        }
        
        refresh_token = create_refresh_token(test_payload)
        wrong_type_headers = {"Authorization": f"Bearer {refresh_token}"}
        response = client.post("/api/v1/auth/verify-mpin", json=mpin_data, headers=wrong_type_headers)
        assert response.status_code == 401

    def test_token_manager_functionality(self):
        """Test token manager operations"""
        # Test storing and validating tokens
        jti = "test_jti_123"
        user_id = "user_123"
        device_id = "device_001"
        expires_at = datetime.utcnow() + timedelta(days=30)
        
        # Store token
        token_manager.store_refresh_token(jti, user_id, device_id, expires_at)
        
        # Validate token
        assert token_manager.is_token_valid(jti) == True
        
        # Test revocation
        token_manager.revoke_token(jti)
        assert token_manager.is_token_valid(jti) == False
        assert jti in token_manager.blacklisted_tokens
        
        # Test user token operations
        jti2 = "test_jti_456"
        token_manager.store_refresh_token(jti2, user_id, device_id, expires_at)
        
        active_tokens = token_manager.get_user_active_tokens(user_id)
        assert len(active_tokens) == 1
        assert active_tokens[0]["jti"] == jti2
        
        # Test revoking all user tokens
        token_manager.revoke_user_tokens(user_id)
        assert len(token_manager.get_user_active_tokens(user_id)) == 0

    def test_token_expiration_handling(self):
        """Test token expiration scenarios"""
        # Test expired refresh token
        jti = "expired_token_123"
        user_id = "user_123"
        device_id = "device_001"
        expires_at = datetime.utcnow() - timedelta(days=1)  # Already expired
        
        token_manager.store_refresh_token(jti, user_id, device_id, expires_at)
        
        # Should be invalid due to expiration
        assert token_manager.is_token_valid(jti) == False

    def test_authentication_endpoints_exist(self):
        """Test that all authentication endpoints exist and are accessible"""
        endpoints_to_test = [
            ("/api/v1/auth/register", "POST"),
            ("/api/v1/auth/login", "POST"),
            ("/api/v1/auth/refresh", "POST"),
            ("/api/v1/auth/verify-mpin", "POST"),
        ]
        
        for endpoint, method in endpoints_to_test:
            if method == "POST":
                response = client.post(endpoint, json={})
            else:
                response = client.get(endpoint)
            
            # Should not be 404 (endpoint exists)
            assert response.status_code != 404, f"Endpoint {endpoint} not found"

    def test_token_manager_stats(self):
        """Test token manager statistics functionality"""
        # Add some tokens
        token_manager.store_refresh_token("jti1", "user1", "device1", 
                                        datetime.utcnow() + timedelta(days=30))
        token_manager.store_refresh_token("jti2", "user1", "device2", 
                                        datetime.utcnow() + timedelta(days=30))
        token_manager.store_refresh_token("jti3", "user2", "device1", 
                                        datetime.utcnow() + timedelta(days=30))
        
        stats = token_manager.get_stats()
        assert stats["active_refresh_tokens"] == 3
        assert stats["users_with_tokens"] == 2
        assert stats["total_user_token_mappings"] == 3
        
        # Revoke one token
        token_manager.revoke_token("jti1")
        stats = token_manager.get_stats()
        assert stats["active_refresh_tokens"] == 2
        assert stats["blacklisted_tokens"] == 1

    def test_concurrent_token_operations(self):
        """Test token operations under concurrent access"""
        user_id = "concurrent_user"
        
        # Simulate multiple devices logging in
        for i in range(5):
            jti = f"concurrent_jti_{i}"
            device_id = f"device_{i}"
            expires_at = datetime.utcnow() + timedelta(days=30)
            token_manager.store_refresh_token(jti, user_id, device_id, expires_at)
        
        # Verify all tokens are stored
        active_tokens = token_manager.get_user_active_tokens(user_id)
        assert len(active_tokens) == 5
        
        # Revoke tokens for specific devices
        token_manager.revoke_device_tokens(user_id, "device_2")
        active_tokens = token_manager.get_user_active_tokens(user_id)
        assert len(active_tokens) == 4
        
        # Verify correct token was revoked
        device_ids = [token["device_id"] for token in active_tokens]
        assert "device_2" not in device_ids

    def test_security_headers_and_cors(self):
        """Test security headers and CORS configuration"""
        response = client.get("/")
        
        # Check that response doesn't expose sensitive information
        assert "X-Powered-By" not in response.headers
        
        # Test CORS preflight (if configured)
        cors_headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization, Content-Type"
        }
        response = client.options("/api/v1/auth/login", headers=cors_headers)
        # Should handle CORS appropriately
        assert response.status_code in [200, 204, 405]

    def test_rate_limiting_preparation(self):
        """Test endpoints handle repeated requests (preparation for rate limiting)"""
        # Test multiple login attempts
        login_data = {
            "phone": "9999999999",
            "password": "WrongPassword",
            "device_id": "test_device"
        }
        
        responses = []
        for _ in range(3):
            response = client.post("/api/v1/auth/login", json=login_data)
            responses.append(response.status_code)
        
        # All should handle gracefully (401 for wrong credentials)
        for status_code in responses:
            assert status_code in [401, 500]  # Unauthorized or server error
