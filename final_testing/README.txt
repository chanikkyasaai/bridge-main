Final Testing Suite for Behavioral Authentication
=================================================

This folder contains all files and scripts for comprehensive, end-to-end testing of the ML engine and behavioral authentication system.

Contents:
---------
- test_user_session_01.json ... test_user_session_11.json: Real user session logs for learning and authentication.
- test_bot_session.json: Simulated bot user session for bot detection.
- new_data.json: Session with device_info and drift scenarios for drift tracker testing.
- test_traitor_session.json: Simulated traitor user with completely different behavioral patterns.
- test_traitor_device_hijack.json: Simulated session with device, IP, and location changes to trigger drift tracker.
- test_ml_engine_comprehensive.py: Main script for comprehensive ML engine testing.
- test_complete_behavioral_flow.py: Script for full behavioral flow testing.
- test_complete_flow_fresh.py: Script for fresh end-to-end flow testing.
- test_backend_behavioral_flow.py: Backend behavioral flow test script.
- test_quick_behavioral.py: Quick behavioral test script.
- test_ml_quick.py: Quick ML test script.
- test_bot_detection.py: Bot detection test script.
- end_to_end_test.py: End-to-end test script.
- test_working_system.py: Working system test script.
- simple_test.py: Simple test script.

How to Use:
-----------
- Use the provided scripts to test all aspects of the behavioral authentication system.
- The runner script (to be created: run_final_test_suite.py) will automate registration, login, session creation, and scenario testing (normal, bot, traitor, hijack).
- Each scenario is designed to test:
  - Normal authentication and learning
  - Bot detection
  - GNN escalation and anomaly detection
  - Behavioral drift tracking
  - Signal routing and audit logging

Add new test scenarios as needed to this folder for future coverage. 