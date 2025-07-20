import random
import time
from components import (
    initialize_driver, primary_login, scan_biometric, get_random_delay,
    view_my_account, add_benificiary, send_money_direct_pay, do_card_less_pay,
    view_transaction_history, view_cards, change_mpin, change_passcode,
    get_mmid, toggle_upi_services
)
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from swipe_down import scroll_down_recommended,scroll_up_recommended
from collections import deque

# User 1 Configuration
Bot_timeout = 2
Bot_timeout_variation = 0
Bot_typing_time_delay = 0
Bot_typing_time_delay_variation = 0
Bot_phone_number = "9444452555"
Bot_password = "12345"
Bot_mpin = "123456"
Bot_scroll_distance=450
Bot_scroll_duration=1000

driver = initialize_driver(timeout=Bot_timeout)
print("Starting Bot simulation script...")

# Always Login
def try_login():
    login_required = True
    try:
        login_button = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((
                AppiumBy.ANDROID_UIAUTOMATOR,
                'new UiSelector().description("Login")'
            ))
        )
        # If found, perform some action
        login_button.click()
        print("Login button clicked.")
        login_required=True
    except Exception as e:
        login_required = False
        print("Login button not found, going to biometric login")

    if login_required:
        try:
            
            
            delay=get_random_delay(center=Bot_typing_time_delay, variation=Bot_typing_time_delay_variation)
            print(f"Typing delay: {delay} seconds")
            timeout=get_random_delay(center=Bot_timeout, variation=Bot_timeout_variation)
            print(f"Timeout: {timeout} seconds")
            primary_login(driver,phone_number=Bot_phone_number,password=Bot_password,timeout=timeout,typingDelay=delay)
            print("Primary login successful.")
        except Exception as e:
            print(f"Primary login failed: {e}")
            driver.quit()
            exit(1)

    ##Biometric Login

    time.sleep(Bot_timeout)
    try:
        timeout=get_random_delay(center=Bot_timeout, variation=Bot_timeout_variation)
        print(f"Timeout for biometric login: {timeout} seconds")
        scan_biometric(driver, timeout=timeout)
        print("Biometric login successful.")
    except Exception as e:
        print(f"Biometric login failed: {e}")
        driver.quit()
        exit(1)

try_login()

# Define optional actions with probabilities and dependencies
actions = [
    {"name": "view_my_account", "func": lambda d, t, delay: view_my_account(d, timeout=t), "prob": 0.9, "dependencies": []},
    {"name": "add_benificiary", "func": lambda d, t, delay: add_benificiary(d, timeout=t, typingDelay=delay), "prob": 0.7, "dependencies": ["view_my_account"]},
    {"name": "send_money_direct_pay", "func": lambda d, t, delay: send_money_direct_pay(d, timeout=t, typingDelay=delay), "prob": 0.7, "dependencies": ["add_benificiary"]},
    {"name": "do_card_less_pay", "func": lambda d, t, delay: do_card_less_pay(d, timeout=t), "prob": 0.6, "dependencies": []},
    {"name": "view_transaction_history", "func": lambda d, t, delay: view_transaction_history(d, timeout=t), "prob": 0.6, "dependencies": ["send_money_direct_pay"]},
    {"name": "view_cards", "func": lambda d, t, delay: view_cards(d, timeout=t), "prob": 0.5, "dependencies": []},
    {"name": "change_mpin", "func": lambda d, t, delay: change_mpin(d, old_mpin=Bot_mpin, new_mpin="654321", timeout=t, typingDelay=delay), "prob": 0.5, "dependencies": ["view_my_account"]},
    {"name": "change_passcode", "func": lambda d, t, delay: change_passcode(d, old_passcode=Bot_password, new_passcode="54321", timeout=t, typingDelay=delay), "prob": 0.5, "dependencies": ["change_mpin"]},
    {"name": "get_mmid", "func": lambda d, t, delay: get_mmid(d, timeout=t), "prob": 0.4, "dependencies": []},
    {"name": "toggle_upi_services", "func": lambda d, t, delay: toggle_upi_services(d, timeout=t), "prob": 0.4, "dependencies": []},
]




# Track executed actions
executed = set()

# Convert actions into a name-to-action map
action_map = {a["name"]: a for a in actions}
pending = deque(actions)  # process one-by-one, like a robot

print("\n--- Starting robotic action execution ---\n")

while pending:
    action = pending.popleft()

    # If already done, skip
    if action["name"] in executed:
        continue

    # Check if dependencies are met
    unmet_deps = [dep for dep in action["dependencies"] if dep not in executed]
    if unmet_deps:
        print(f"Delaying '{action['name']}' due to unmet dependencies: {unmet_deps}")
        pending.append(action)  # push back to queue
        continue

    try:
        timeout = get_random_delay(center=Bot_timeout, variation=Bot_timeout_variation)
        delay = get_random_delay(center=Bot_typing_time_delay, variation=Bot_typing_time_delay_variation)
        print(f"🔧 Executing: {action['name']} (timeout={timeout:.2f}, delay={delay:.2f})")
        action["func"](driver, timeout, delay)
        executed.add(action["name"])
        time.sleep(get_random_delay(0.6, 0.15))  # robotic delay between actions
    except Exception as e:
        print(f"❌ Failed action: {action['name']} -> {e}")
        driver.quit()
        exit(1)

print("\n✅ Bot simulation completed successfully.\n")
driver.quit()

print("Bot simulation completed successfully.")
driver.quit()
