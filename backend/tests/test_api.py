import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "Security Backend" in data["message"]

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_register_user():
    """Test user registration with MPIN"""
    user_data = {
        "email": "test@example.com",
        "password": "testpassword",
        "mpin": "1234"
    }
    
    response = client.post("/api/v1/auth/register", json=user_data)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == user_data["email"]
    assert "message" in data

def test_register_invalid_mpin():
    """Test registration with invalid MPIN"""
    user_data = {
        "email": "test2@example.com", 
        "password": "testpassword",
        "mpin": "12345"  # Invalid - too long
    }
    
    response = client.post("/api/v1/auth/register", json=user_data)
    assert response.status_code == 400

def test_login_user():
    """Test user login and session creation"""
    # First register a user
    user_data = {
        "email": "testlogin@example.com",
        "password": "testpassword",
        "mpin": "1234"
    }
    client.post("/api/v1/auth/register", json=user_data)
    
    # Then try to login
    login_data = {
        "email": "testlogin@example.com",
        "password": "testpassword",
        "device_id": "test_device"
    }
    
    response = client.post("/api/v1/auth/login", json=login_data)
    assert response.status_code == 200
    data = response.json()
    assert "session_token" in data
    assert "session_id" in data
    assert data["token_type"] == "bearer"

def test_mpin_verification():
    """Test MPIN verification"""
    # Register and login user
    email = "testmpin@example.com"
    password = "testpassword"
    mpin = "1234"
    
    # Register
    client.post("/api/v1/auth/register", json={
        "email": email,
        "password": password,
        "mpin": mpin
    })
    
    # Login
    login_response = client.post("/api/v1/auth/login", json={
        "email": email,
        "password": password,
        "device_id": "test_device"
    })
    token = login_response.json()["session_token"]
    
    # Verify MPIN
    headers = {"Authorization": f"Bearer {token}"}
    mpin_data = {"mpin": mpin}
    
    response = client.post("/api/v1/auth/verify-mpin", json=mpin_data, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "verified"

def test_invalid_mpin_verification():
    """Test invalid MPIN verification"""
    # Register and login user
    email = "testinvalidmpin@example.com"
    password = "testpassword"
    mpin = "1234"
    
    # Register
    client.post("/api/v1/auth/register", json={
        "email": email,
        "password": password,
        "mpin": mpin
    })
    
    # Login
    login_response = client.post("/api/v1/auth/login", json={
        "email": email,
        "password": password,
        "device_id": "test_device"
    })
    token = login_response.json()["session_token"]
    
    # Try invalid MPIN
    headers = {"Authorization": f"Bearer {token}"}
    mpin_data = {"mpin": "9999"}  # Wrong MPIN
    
    response = client.post("/api/v1/auth/verify-mpin", json=mpin_data, headers=headers)
    assert response.status_code == 401

def test_session_status():
    """Test getting session status"""
    # Register and login user
    email = "teststatus@example.com"
    password = "testpassword"
    mpin = "1234"
    
    # Register
    client.post("/api/v1/auth/register", json={
        "email": email,
        "password": password,
        "mpin": mpin
    })
    
    # Login
    login_response = client.post("/api/v1/auth/login", json={
        "email": email,
        "password": password,
        "device_id": "test_device"
    })
    token = login_response.json()["session_token"]
    
    # Get session status
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/api/v1/auth/session-status", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "risk_score" in data
    assert "is_active" in data
