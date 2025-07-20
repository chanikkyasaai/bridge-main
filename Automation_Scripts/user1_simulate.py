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

def biased_random_bool():
    return random.choices([True, False], weights=[1, 2])[0]
# User 1 Configuration
user1_timeout = 2
user1_timeout_variation = 0.5
user1_typing_time_delay = 0.5
user1_typing_time_delay_variation = 0.2
user1_phone_number = "9444452555"
user1_password = "12345"
user1_mpin = "123456"
user1_scroll_distance=450
user1_scroll_duration=1000

driver = initialize_driver(timeout=user1_timeout)
print("Starting user1 simulation script...")

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
            flag=biased_random_bool()
            if flag:
                scroll_down_recommended(driver,distance=user1_scroll_distance,duration=user1_scroll_duration)
                scroll_up_recommended(driver,distance=user1_scroll_distance,duration=user1_scroll_duration)
            delay=get_random_delay(center=user1_typing_time_delay, variation=user1_typing_time_delay_variation)
            print(f"Typing delay: {delay} seconds")
            timeout=get_random_delay(center=user1_timeout, variation=user1_timeout_variation)
            print(f"Timeout: {timeout} seconds")
            primary_login(driver,phone_number=user1_phone_number,password=user1_password,timeout=timeout,typingDelay=delay)
            print("Primary login successful.")
        except Exception as e:
            print(f"Primary login failed: {e}")
            driver.quit()
            exit(1)

    ##Biometric Login

    time.sleep(user1_timeout)
    try:
        timeout=get_random_delay(center=user1_timeout, variation=user1_timeout_variation)
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
    {"name": "change_mpin", "func": lambda d, t, delay: change_mpin(d, old_mpin=user1_mpin, new_mpin="654321", timeout=t, typingDelay=delay), "prob": 0.5, "dependencies": ["view_my_account"]},
    {"name": "change_passcode", "func": lambda d, t, delay: change_passcode(d, old_passcode=user1_password, new_passcode="54321", timeout=t, typingDelay=delay), "prob": 0.5, "dependencies": ["change_mpin"]},
    {"name": "get_mmid", "func": lambda d, t, delay: get_mmid(d, timeout=t), "prob": 0.4, "dependencies": []},
    {"name": "toggle_upi_services", "func": lambda d, t, delay: toggle_upi_services(d, timeout=t), "prob": 0.4, "dependencies": []},
]


# Track executed actions
executed = set()

# Run random actions based on probabilities and dependencies
random.shuffle(actions)
for action in actions:
    if random.random() <= action["prob"]:
        if all(dep in executed for dep in action["dependencies"]):
            try:
                timeout = get_random_delay(center=user1_timeout, variation=user1_timeout_variation)
                delay = get_random_delay(center=user1_typing_time_delay, variation=user1_typing_time_delay_variation)
                print(f"Running action: {action['name']} with timeout={timeout:.2f}, delay={delay:.2f}")
                action["func"](driver, timeout, delay)
                executed.add(action["name"])
                time.sleep(get_random_delay(0.5, 0.2))
            except Exception as e:
                print(f"Failed action: {action['name']} -> {e}")
                driver.quit()
                exit(1)
        else:
            print(f"Skipping {action['name']} due to unmet dependencies.")
    else:
        print(f"Skipping {action['name']} randomly.")
print("User1 simulation completed successfully.")
driver.quit()
