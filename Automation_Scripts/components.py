from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
from swipe_down import scroll_down_recommended
def createCapabilities():
    # Create capabilities using AppiumOptions
    caps = UiAutomator2Options().load_capabilities({
        "platformName": "Android",
        "deviceName": "Android Device",
        "automationName": "UiAutomator2",
        "appPackage": "com.example.canara_ai",
        "appActivity": "com.example.canara_ai.MainActivity",
        "noReset": True,
        "enforceAppInstall": False,
        "disableSuppressAccessibilityService": True,
        "ignoreHiddenApiPolicyError": True
    })
    return caps

def initialize_driver(timeout=None):
    # Create capabilities
    caps = createCapabilities()
    
    # Initialize the Appium driver
    driver = webdriver.Remote("http://127.0.0.1:4723", options=caps)
    if(timeout):
        time.sleep(timeout)
    return driver

def login_phone_number_element(driver,phone_number="9444452444",timeout=None):
    phone_number_element=driver.find_element(
    AppiumBy.ANDROID_UIAUTOMATOR,
    'new UiSelector().className("android.widget.EditText").instance(0)'
    )
    phone_number_element.click()  # First tap to focus
    if timeout:
        time.sleep(timeout)
    phone_number_element.send_keys(phone_number)

def driver_hide_keyboard(driver, timeout=None):
    driver.hide_keyboard()
    if timeout:
        time.sleep(timeout)

def login_password_element(driver, password="12345", timeout=None):
    password_element = driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,
        'new UiSelector().className("android.widget.EditText").instance(1)'
    )
    password_element.click()  # First tap to focus
    if timeout:
        time.sleep(timeout)
    password_element.send_keys(password)
    driver_hide_keyboard(driver,timeout=timeout)
    
def login_button_click(driver):
    login_button = driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,
        'new UiSelector().description("Login")'
    )
    login_button.click()

def go_to_registration_page(driver,timeout=None):
    driver.find_element(AppiumBy.ACCESSIBILITY_ID, "Don't have an account? Register").click()
    if timeout:
        time.sleep(timeout)

def fill_registration_form(driver, phone_number="9444452444", password="12345", confirm_password="12345",timeout=None):
    phone_number_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(0)')
    phone_number_field.click()  # First tap to focus
    phone_number_field.send_keys(phone_number)
    driver_hide_keyboard(driver,timeout=timeout)

    print("Entering password...")
    password_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,
        'new UiSelector().className("android.widget.EditText").instance(1)'
    )
    password_field.click()  # First tap to focus
    password_field.send_keys("12345")
    driver_hide_keyboard(driver,timeout=timeout)

    print("Confirming password...")
    confirm_password_field =driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,
        'new UiSelector().className("android.widget.EditText").instance(2)'
    )

    confirm_password_field.click()  # First tap to focus
    confirm_password_field.send_keys("12345")
    driver_hide_keyboard(driver,timeout=timeout)

    driver.find_element(AppiumBy.ACCESSIBILITY_ID, "Register").click()
    if timeout:
        time.sleep(timeout)
def close_driver(driver):
    if driver:
        driver.quit()
        print("Driver closed successfully.")
    else:
        print("No driver to close.")

def secondary_login_button(driver,pin="12345",timeout=None):
    secondary_login_pin=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.view.View").instance(7)')
    secondary_login_pin.click()
    if timeout:
        time.sleep(timeout)
    secondary_login_pin.send_keys(pin)
    driver_hide_keyboard(driver,timeout=timeout)

def scan_biometric(driver,timeout=None):
    fingerprint_button=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.view.View").instance(8)'
    )
    fingerprint_button.click()
    if timeout:
        time.sleep(timeout)
    
    login_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Login'
    )
    login_button.click()
    print("Biometric scan initiated and passed!!!.")
    if timeout:
        time.sleep(timeout)

def add_benificiary(driver,name="Deez Nuts", upi_id="123456789@upi",mobile_number="1234567890", timeout=None):
    add_benificiary_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Add Beneficiary'
    )
    add_benificiary_button.click()
    if timeout:
        time.sleep(timeout)
    
    name_field=driver.find_element(AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(0)')
    name_field.click()
    name_field.send_keys(name)


    upi_id_field=driver.find_element(AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(1)')
    upi_id_field.click()
    upi_id_field.send_keys(upi_id)
   
    mobile_number_field=driver.find_element(AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(2)')
    mobile_number_field.click() 
    mobile_number_field.send_keys(mobile_number)
    

    add_button=driver.find_element(AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().description("Add")')
    add_button.click()
    print("Beneficiary added successfully.")
def send_money(driver,timeout=None):
    scroll_down_recommended(driver)
    send_money_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Send Money'
    )
    send_money_button.click()
    send_money_button.click()  # Click again to ensure the action is registered
    if timeout:
        time.sleep(timeout)
    print("Navigated to Send Money page.")
    add_benificiary(driver,timeout=timeout)


def send_money_direct_pay(driver,upi_id="9444524432@yaml",amount="6969",timeout=None):
    
    direct_pay_btn=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,"Direct Pay"
    )
    direct_pay_btn.click()
    if timeout:
        time.sleep(timeout)
    upi_id_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(0)'
    )
    upi_id_field.click()
    upi_id_field.send_keys(upi_id)
    

    amount_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(1)'
    )
    amount_field.click()
    amount_field.send_keys(amount)
   

    pay_btn=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().description("Pay")'
    )
    pay_btn.click()
    print("Payment initiated successfully.")
    if timeout:
        time.sleep(timeout)


def direct_pay(driver,account_number="124385490367",beneficiary_name="Deez Nuts",Nick_name="earpods",timeout=None):
    direct_pay_btn=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,"Direct Pay"
    )
    direct_pay_btn.click()
    if timeout:
        time.sleep(timeout)
    
    account_number_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(0)'
    )
    account_number_field.click()
    account_number_field.send_keys(account_number)

    reenter_account_number_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(1)'
    )
    reenter_account_number_field.click()
    reenter_account_number_field.send_keys(account_number)

    beneficiary_name_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(2)'
    )
    beneficiary_name_field.click()
    beneficiary_name_field.send_keys(beneficiary_name)

    nick_name_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(3)'
    )
    nick_name_field.click()
    nick_name_field.send_keys(Nick_name)

    driver_hide_keyboard(driver, timeout=timeout)

    pay_btn=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().description("Confirm")'
    )
    pay_btn.click()
    
    wait=WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((AppiumBy.ANDROID_UIAUTOMATOR, 'new UiSelector().description("Back")')))
    back_button=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().description("Back")'
    )
    back_button.click()
    print("Direct Pay initiated successfully.")

def open_deposit(driver,full_name="Deez Nuts",mobile_number="1234567890",timeout=None):
    #open_deposit_button=driver.find_element(
    #    AppiumBy.ACCESSIBILITY_ID,'Open Deposit'
    #)
    #open_deposit_button.click()
    wait=WebDriverWait(driver, timeout if timeout else 10)
    wait.until(EC.presence_of_element_located((AppiumBy.ANDROID_UIAUTOMATOR, 'new UiSelector().className("android.widget.EditText").instance(0)')))
    full_name_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(0)'
    )
    full_name_field.click()
    full_name_field.send_keys(full_name)
    driver_hide_keyboard(driver, timeout=timeout)
    mobile_number_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(1)'
    )
    mobile_number_field.click()
    mobile_number_field.send_keys(mobile_number)

    driver_hide_keyboard(driver, timeout=timeout)

    select_bank=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Select Bank'
    )
    select_bank.click()
    wait.until(EC.presence_of_element_located((AppiumBy.ANDROID_UIAUTOMATOR, 'new UiSelector().description("State Bank of India")')))

    bank_option=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().description("State Bank of India")'
    )
    bank_option.click()
    if timeout:
        time.sleep(timeout)
    register_upi=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().description("Register UPI")'
    )
    register_upi.click()
    
def go_to_profile(driver,timeout=None):
    profile_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Profile'
    )
    profile_button.click()
    if timeout:
        time.sleep(timeout)
    
def change_mpin(driver,old_mpin="12345",new_mpin="54321",timeout=None):
    change_mpin_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Change MPIN'
    )
    change_mpin_button.click()
    wait=WebDriverWait(driver, timeout if timeout else 10)
    wait.until(EC.presence_of_element_located((AppiumBy.ANDROID_UIAUTOMATOR, 'new UiSelector().className("android.widget.EditText").instance(0)')))
    old_mpin_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(0)'
    )
    old_mpin_field.click()
    old_mpin_field.send_keys(old_mpin)
    new_mpin_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(1)'
    )
    new_mpin_field.click()
    new_mpin_field.send_keys(new_mpin)

    change_Btn=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Cancel'
    )
    change_Btn.click()
    print("MPIN changed successfully.")

def goto_bank(driver,timeout=None):
    bank_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Bank'
    )
    bank_button.click()
    if timeout:
        time.sleep(timeout)
    print("Navigated to Bank section.")

def do_card_less_pay(driver,amount="6969",mobile_number="1123456789",timeout=None):
    card_less_pay_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Card-less Cash'
    )
    card_less_pay_button.click()
    wait=WebDriverWait(driver, timeout if timeout else 10)
    wait.until(EC.presence_of_element_located((AppiumBy.ANDROID_UIAUTOMATOR, 'new UiSelector().className("android.widget.EditText").instance(0)')))

    amount_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(0)'
    )
    amount_field.click()
    amount_field.send_keys(amount)
    

    mobile_number_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(1)'
    )   
    mobile_number_field.click()
    mobile_number_field.send_keys(mobile_number)
    driver_hide_keyboard(driver, timeout=timeout)

    confirm_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Generate Cash Code'
    ) 
    confirm_button.click()
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, 'OK')))
    ok_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'OK'
    )
    ok_button.click()

    wait.until(EC.presence_of_element_located((AppiumBy.ANDROID_UIAUTOMATOR, 'new UiSelector().className("android.widget.Button").instance(0)')))
    back_button=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.Button").instance(0)'
    )   
    back_button.click()
    if timeout:
        time.sleep(timeout)
    print("Cardless Pay initiated successfully.")