from components import initialize_driver, secondary_login_button,scan_biometric

driver=initialize_driver(timeout=2)
print("Starting secondary login script...")
print("Waiting for secondary login button...")
secondary_login_button(driver,timeout=5,typingDelay=0.2)
print("Scanning biometric...")
scan_biometric(driver, timeout=5)