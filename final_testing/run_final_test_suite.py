import requests
import json
import random
import string
import time
from pathlib import Path

# Backend endpoints (adjust if needed)
BACKEND_URL = "http://127.0.0.1:8001"
API_AUTH_URL = "http://127.0.0.1:8000/api/v1/auth"
REGISTER_ENDPOINT = f"{API_AUTH_URL}/register"
LOGIN_ENDPOINT = f"{API_AUTH_URL}/login"
SESSION_START_ENDPOINT = f"{BACKEND_URL}/session/start"
SESSION_END_ENDPOINT = f"{BACKEND_URL}/session/end"
ANALYZE_ENDPOINT = f"{BACKEND_URL}/analyze-mobile"

# Test data files
USER_SESSION_FILES = [f"final_testing/test_user_session_{i:02d}.json" for i in range(1, 12)]
BOT_SESSION_FILE = "final_testing/test_bot_session.json"
TRAITOR_SESSION_FILE = "final_testing/test_traitor_session.json"
HIJACK_SESSION_FILE = "final_testing/test_traitor_device_hijack.json"

# Registration constraints
MPIN_LENGTH = 5  # Must match backend

# Utility functions
def random_phone():
    return "9" + "".join(random.choices(string.digits, k=9))

def random_password():
    return "Test@" + "".join(random.choices(string.ascii_letters + string.digits, k=8))

def random_mpin():
    return "".join(random.choices(string.digits, k=MPIN_LENGTH))

def random_device_id():
    return "device-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def print_result(label, resp):
    print(f"\n--- {label} ---")
    print(f"Status: {resp.status_code}")
    try:
        print(json.dumps(resp.json(), indent=2))
    except Exception:
        print(resp.text)

def run_final_test_suite():
    print("=== Final Behavioral Authentication Test Suite ===")
    # 1. Register new user
    phone = random_phone()
    password = random_password()
    mpin = random_mpin()
    device_id = random_device_id()
    print(f"Registering user: phone={phone}, password={password}, mpin={mpin}, device_id={device_id}")
    reg_payload = {"phone": phone, "password": password, "mpin": mpin}
    reg_resp = requests.post(REGISTER_ENDPOINT, json=reg_payload)
    print_result("Registration", reg_resp)
    if reg_resp.status_code != 200:
        print("❌ Registration failed. Exiting.")
        return
    # 2. Login
    login_payload = {"phone": phone, "password": password, "device_id": device_id}
    login_resp = requests.post(LOGIN_ENDPOINT, json=login_payload)
    print_result("Login", login_resp)
    if login_resp.status_code != 200:
        print("❌ Login failed. Exiting.")
        return
    access_token = login_resp.json().get("access_token")
    headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
    # 3. Simulate 10 normal sessions
    for i, session_file in enumerate(USER_SESSION_FILES, 1):
        session_data = load_json(session_file)
        session_id = session_data["session_id"] if isinstance(session_data, dict) else session_data[0]["session_id"]
        print(f"\n=== Normal Session {i} ({session_id}) ===")
        # Start session
        start_payload = {"user_id": session_data["user_id"], "session_id": session_id, "device_info": {"device_id": device_id}}
        start_resp = requests.post(SESSION_START_ENDPOINT, json=start_payload, headers=headers)
        print_result(f"Session {i} Start", start_resp)
        # Send logs
        analyze_payload = {"user_id": session_data["user_id"], "session_id": session_id, "logs": session_data["logs"]}
        analyze_resp = requests.post(ANALYZE_ENDPOINT, json=analyze_payload, headers=headers)
        print_result(f"Session {i} Analyze", analyze_resp)
        # End session
        end_payload = {"session_id": session_id, "reason": "completed"}
        end_resp = requests.post(SESSION_END_ENDPOINT, json=end_payload, headers=headers)
        print_result(f"Session {i} End", end_resp)
        time.sleep(0.5)
    # 4. Simulate bot session
    print("\n=== Simulating Bot Session ===")
    bot_data = load_json(BOT_SESSION_FILE)
    bot_session_id = bot_data["session_id"]
    start_payload = {"user_id": bot_data["user_id"], "session_id": bot_session_id, "device_info": {"device_id": "bot-device-001"}}
    start_resp = requests.post(SESSION_START_ENDPOINT, json=start_payload, headers=headers)
    print_result("Bot Session Start", start_resp)
    analyze_payload = {"user_id": bot_data["user_id"], "session_id": bot_session_id, "logs": bot_data["logs"]}
    analyze_resp = requests.post(ANALYZE_ENDPOINT, json=analyze_payload, headers=headers)
    print_result("Bot Session Analyze", analyze_resp)
    end_payload = {"session_id": bot_session_id, "reason": "completed"}
    end_resp = requests.post(SESSION_END_ENDPOINT, json=end_payload, headers=headers)
    print_result("Bot Session End", end_resp)
    # 5. Simulate traitor session
    print("\n=== Simulating Traitor Session ===")
    traitor_data = load_json(TRAITOR_SESSION_FILE)
    traitor_session_id = traitor_data["session_id"]
    start_payload = {"user_id": traitor_data["user_id"], "session_id": traitor_session_id, "device_info": {"device_id": "traitor-device-001"}}
    start_resp = requests.post(SESSION_START_ENDPOINT, json=start_payload, headers=headers)
    print_result("Traitor Session Start", start_resp)
    analyze_payload = {"user_id": traitor_data["user_id"], "session_id": traitor_session_id, "logs": traitor_data["logs"]}
    analyze_resp = requests.post(ANALYZE_ENDPOINT, json=analyze_payload, headers=headers)
    print_result("Traitor Session Analyze", analyze_resp)
    end_payload = {"session_id": traitor_session_id, "reason": "completed"}
    end_resp = requests.post(SESSION_END_ENDPOINT, json=end_payload, headers=headers)
    print_result("Traitor Session End", end_resp)
    # 6. Simulate hijack session
    print("\n=== Simulating Device Hijack Session ===")
    hijack_data = load_json(HIJACK_SESSION_FILE)
    hijack_session_id = hijack_data["session_id"]
    start_payload = {"user_id": hijack_data["user_id"], "session_id": hijack_session_id, "device_info": {"device_id": "traitor-device-002"}}
    start_resp = requests.post(SESSION_START_ENDPOINT, json=start_payload, headers=headers)
    print_result("Hijack Session Start", start_resp)
    analyze_payload = {"user_id": hijack_data["user_id"], "session_id": hijack_session_id, "logs": hijack_data["logs"]}
    analyze_resp = requests.post(ANALYZE_ENDPOINT, json=analyze_payload, headers=headers)
    print_result("Hijack Session Analyze", analyze_resp)
    end_payload = {"session_id": hijack_session_id, "reason": "completed"}
    end_resp = requests.post(SESSION_END_ENDPOINT, json=end_payload, headers=headers)
    print_result("Hijack Session End", end_resp)
    print("\n=== Test Suite Complete ===")

if __name__ == "__main__":
    run_final_test_suite() 