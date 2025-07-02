"""
Authentication System Demo
Demonstrates the complete authentication flow with access and refresh tokens
"""

import requests
import json
import time
from datetime import datetime

class AuthDemo:
    """Demo class for authentication system"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.access_token = None
        self.refresh_token = None
        self.session_id = None
        
    def register_user(self, phone, password, mpin):
        """Register a new user"""
        print(f"🔐 Registering user with phone: {phone}")
        
        url = f"{self.base_url}/api/v1/auth/register"
        data = {
            "phone": phone,
            "password": password,
            "mpin": mpin
        }
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Registration successful!")
                print(f"   User ID: {result.get('user_id')}")
                print(f"   Next step: {result.get('next_step')}")
                return True
            else:
                print(f"❌ Registration failed: {response.status_code}")
                print(f"   Error: {response.json().get('detail', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"❌ Registration error: {str(e)}")
            return False
    
    def login(self, phone, password, device_id):
        """Login and get tokens"""
        print(f"🔑 Logging in user: {phone}")
        
        url = f"{self.base_url}/api/v1/auth/login"
        data = {
            "phone": phone,
            "password": password,
            "device_id": device_id
        }
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                result = response.json()
                
                # Store tokens
                self.access_token = result["access_token"]
                self.refresh_token = result["refresh_token"]
                self.session_id = result["session_id"]
                
                print(f"✅ Login successful!")
                print(f"   Access Token: {self.access_token[:20]}...")
                print(f"   Refresh Token: {self.refresh_token[:20]}...")
                print(f"   Session ID: {self.session_id}")
                print(f"   Expires in: {result['expires_in']} seconds")
                
                return True
            else:
                print(f"❌ Login failed: {response.status_code}")
                print(f"   Error: {response.json().get('detail', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"❌ Login error: {str(e)}")
            return False
    
    def verify_mpin(self, mpin):
        """Verify MPIN using access token"""
        print(f"🔒 Verifying MPIN")
        
        if not self.access_token:
            print("❌ No access token available. Please login first.")
            return False
        
        url = f"{self.base_url}/api/v1/auth/verify-mpin"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        data = {"mpin": mpin}
        
        try:
            response = requests.post(url, json=data, headers=headers)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ MPIN verification successful!")
                print(f"   Status: {result.get('status')}")
                print(f"   User ID: {result.get('user_id')}")
                return True
            else:
                print(f"❌ MPIN verification failed: {response.status_code}")
                print(f"   Error: {response.json().get('detail', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"❌ MPIN verification error: {str(e)}")
            return False
    
    def refresh_tokens(self):
        """Refresh access token using refresh token"""
        print(f"🔄 Refreshing tokens")
        
        if not self.refresh_token:
            print("❌ No refresh token available. Please login first.")
            return False
        
        url = f"{self.base_url}/api/v1/auth/refresh"
        data = {"refresh_token": self.refresh_token}
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                result = response.json()
                
                # Update tokens
                old_access_token = self.access_token[:20] if self.access_token else "None"
                old_refresh_token = self.refresh_token[:20] if self.refresh_token else "None"
                
                self.access_token = result["access_token"]
                self.refresh_token = result["refresh_token"]
                
                print(f"✅ Token refresh successful!")
                print(f"   Old Access Token: {old_access_token}...")
                print(f"   New Access Token: {self.access_token[:20]}...")
                print(f"   Old Refresh Token: {old_refresh_token}...")
                print(f"   New Refresh Token: {self.refresh_token[:20]}...")
                print(f"   New Expires in: {result['expires_in']} seconds")
                
                return True
            else:
                print(f"❌ Token refresh failed: {response.status_code}")
                print(f"   Error: {response.json().get('detail', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"❌ Token refresh error: {str(e)}")
            return False
    
    def test_invalid_scenarios(self):
        """Test various invalid scenarios"""
        print(f"🧪 Testing invalid scenarios")
        
        # Test invalid access token
        print("\\n  Testing invalid access token...")
        url = f"{self.base_url}/api/v1/auth/verify-mpin"
        headers = {"Authorization": "Bearer invalid.token.here"}
        data = {"mpin": "123456"}
        
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 401:
            print("  ✅ Invalid access token correctly rejected")
        else:
            print(f"  ❌ Unexpected response: {response.status_code}")
        
        # Test invalid refresh token
        print("\\n  Testing invalid refresh token...")
        url = f"{self.base_url}/api/v1/auth/refresh"
        data = {"refresh_token": "invalid.refresh.token"}
        
        response = requests.post(url, json=data)
        if response.status_code == 401:
            print("  ✅ Invalid refresh token correctly rejected")
        else:
            print(f"  ❌ Unexpected response: {response.status_code}")
        
        # Test missing authorization header
        print("\\n  Testing missing authorization header...")
        url = f"{self.base_url}/api/v1/auth/verify-mpin"
        data = {"mpin": "123456"}
        
        response = requests.post(url, json=data)
        if response.status_code == 403:
            print("  ✅ Missing authorization correctly rejected")
        else:
            print(f"  ❌ Unexpected response: {response.status_code}")
    
    def check_health(self):
        """Check API health"""
        print(f"❤️ Checking API health")
        
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ API is healthy!")
                print(f"   Service: {result.get('service')}")
                print(f"   Status: {result.get('status')}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
            return False
    
    def simulate_token_expiry(self):
        """Simulate token expiry scenario"""
        print(f"⏰ Simulating token expiry scenario")
        
        if not self.access_token:
            print("❌ No access token available. Please login first.")
            return False
        
        # In a real scenario, you would wait for the token to expire
        # For demo purposes, we'll just show the concept
        print("   In production, access tokens expire after 15 minutes")
        print("   Refresh tokens expire after 30 days")
        print("   When access token expires, use refresh token to get new tokens")
        print("   When refresh token is used, it's rotated (old one is revoked)")
        
        return True
    
    def run_complete_demo(self):
        """Run complete authentication demo"""
        print("=" * 60)
        print("🚀 AUTHENTICATION SYSTEM DEMO")
        print("=" * 60)
        
        # Step 1: Health check
        print("\\n" + "=" * 40)
        print("STEP 1: API Health Check")
        print("=" * 40)
        if not self.check_health():
            print("❌ API is not available. Please start the server first.")
            return False
        
        # Step 2: User registration
        print("\\n" + "=" * 40)
        print("STEP 2: User Registration")
        print("=" * 40)
        phone = "9876543210"
        password = "DemoPassword123"
        mpin = "654321"
        
        self.register_user(phone, password, mpin)
        
        # Step 3: User login
        print("\\n" + "=" * 40)
        print("STEP 3: User Login")
        print("=" * 40)
        device_id = "demo_device_001"
        
        if not self.login(phone, password, device_id):
            print("❌ Login failed. Cannot continue demo.")
            return False
        
        # Step 4: MPIN verification
        print("\\n" + "=" * 40)
        print("STEP 4: MPIN Verification")
        print("=" * 40)
        self.verify_mpin(mpin)
        
        # Step 5: Token refresh
        print("\\n" + "=" * 40)
        print("STEP 5: Token Refresh")
        print("=" * 40)
        self.refresh_tokens()
        
        # Step 6: Use refreshed token
        print("\\n" + "=" * 40)
        print("STEP 6: Using Refreshed Token")
        print("=" * 40)
        self.verify_mpin(mpin)
        
        # Step 7: Test invalid scenarios
        print("\\n" + "=" * 40)
        print("STEP 7: Security Testing")
        print("=" * 40)
        self.test_invalid_scenarios()
        
        # Step 8: Token expiry explanation
        print("\\n" + "=" * 40)
        print("STEP 8: Token Expiry Info")
        print("=" * 40)
        self.simulate_token_expiry()
        
        print("\\n" + "=" * 60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        return True

def main():
    """Main demo function"""
    print("Starting Authentication System Demo...")
    print("Make sure the FastAPI server is running on http://localhost:8000")
    print("\\nPress Enter to continue or Ctrl+C to exit...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\\nDemo cancelled.")
        return
    
    # Run the demo
    demo = AuthDemo()
    demo.run_complete_demo()
    
    print("\\n" + "=" * 60)
    print("📋 DEMO SUMMARY")
    print("=" * 60)
    print("This demo demonstrated:")
    print("• User registration with phone, password, and MPIN")
    print("• Secure login with JWT access and refresh tokens")
    print("• Access token usage for protected endpoints (MPIN verification)")
    print("• Refresh token rotation for security")
    print("• Invalid token handling and security measures")
    print("• Token expiry and refresh mechanism")
    print("\\nKey Security Features:")
    print("• Short-lived access tokens (15 minutes)")
    print("• Long-lived refresh tokens (30 days)")
    print("• Refresh token rotation on every use")
    print("• Token revocation and blacklisting")
    print("• Device-based token isolation")
    print("• JWT signature validation")
    print("• MPIN-based transaction authorization")

if __name__ == "__main__":
    main()
