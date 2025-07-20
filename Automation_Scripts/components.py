from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import time
from swipe_down import scroll_down_recommended,scroll_up_recommended
import random

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

def slowTyping(element,text,delay=0.1):
    """
    Types text into an element with a delay between each character.
    
    :param element: The WebElement to type into.
    :param text: The text to type.
    :param delay: Delay in seconds between each character.
    """
    
    combined_text = ""
    for char in text:
        
        element.send_keys(char)
       
        time.sleep(delay)
    element.send_keys(text)
def initialize_driver(timeout=None):
    # Create capabilities
    caps = createCapabilities()
    
    # Initialize the Appium driver
    driver = webdriver.Remote("http://127.0.0.1:4723", options=caps)
    if(timeout):
        time.sleep(timeout)
    return driver

def login_phone_number_element(driver,phone_number="9444452444",timeout=None,typingDelay=0.1):
    phone_number_element=driver.find_element(
    AppiumBy.ANDROID_UIAUTOMATOR,
    'new UiSelector().className("android.widget.EditText").instance(0)'
    )
    phone_number_element.click()  # First tap to focus
    if timeout:
        time.sleep(timeout)
    delay = typingDelay if typingDelay is not None else 0.1
    slowTyping(phone_number_element, phone_number, delay)
    

def driver_hide_keyboard(driver, timeout=None):
    driver.hide_keyboard()
    if timeout:
        time.sleep(timeout)

def login_password_element(driver, password="12345", timeout=None,tyingDelay=0.1):
    password_element = driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,
        'new UiSelector().className("android.widget.EditText").instance(1)'
    )
    password_element.click()  # First tap to focus
    if timeout:
        time.sleep(timeout)
    delay = tyingDelay if tyingDelay is not None else 0.1
    slowTyping(password_element, password, delay)
    
    driver_hide_keyboard(driver,timeout=timeout)
    
def login_button_click(driver):
    login_button = driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,
        'new UiSelector().description("Login")'
    )
    login_button.click()

def primary_login(driver,phone_number="9444452444",password="12345",timeout=None,typingDelay=0.1):
    login_phone_number_element(driver, phone_number=phone_number, timeout=timeout, typingDelay=typingDelay)
    driver_hide_keyboard(driver, timeout=timeout)
    login_password_element(driver, password=password, timeout=timeout, tyingDelay=typingDelay)
    login_button_click(driver)
    if timeout:
        time.sleep(timeout)


def go_to_registration_page(driver,timeout=None):
    driver.find_element(AppiumBy.ACCESSIBILITY_ID, "Don't have an account? Register").click()
    if timeout:
        time.sleep(timeout)

def fill_registration_form(driver, phone_number="9444452444", password="12345", confirm_password="12345",timeout=None,typingDelay=0.1):
    phone_number_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(0)')
    phone_number_field.click()  # First tap to focus
    delay= typingDelay if typingDelay is not None else 0.1
    slowTyping(phone_number_field, phone_number, delay)
    
    driver_hide_keyboard(driver,timeout=timeout)

    print("Entering password...")
    password_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,
        'new UiSelector().className("android.widget.EditText").instance(1)'
    )
    password_field.click()  # First tap to focus
    slowTyping(password_field, password, delay=delay)
    driver_hide_keyboard(driver,timeout=timeout)

    print("Confirming password...")
    confirm_password_field =driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,
        'new UiSelector().className("android.widget.EditText").instance(2)'
    )

    confirm_password_field.click()  # First tap to focus
    slowTyping(confirm_password_field, confirm_password, delay=delay)
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

def secondary_login_button(driver,pin="12345",timeout=None,typingDelay=0.1):
    secondary_login_pin=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.view.View").instance(7)')
    secondary_login_pin.click()
    if timeout:
        time.sleep(timeout)
    delay= typingDelay if typingDelay is not None else 0.1
    slowTyping(secondary_login_pin, pin, delay=delay)
    
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

def add_benificiary(driver,name="Deez Nuts", upi_id="123456789@upi",mobile_number="1234567890", timeout=None,typingDelay=0.1):
    send_money(driver, timeout=timeout)
    typingDelay = typingDelay if typingDelay is not None else 0.1
    wait=WebDriverWait(driver, timeout if timeout else 10)
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, "Add Beneficiary")))
    add_benificiary_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Add Beneficiary'
    )
    add_benificiary_button.click()
    if timeout:
        time.sleep(timeout)

    name_field=driver.find_element(AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(0)')
    name_field.click()
    slowTyping(name_field, name, delay=typingDelay)
    


    upi_id_field=driver.find_element(AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(1)')
    upi_id_field.click()
    slowTyping(upi_id_field, upi_id, delay=typingDelay)
    
   
    mobile_number_field=driver.find_element(AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(2)')
    mobile_number_field.click() 
    slowTyping(mobile_number_field, mobile_number, delay=typingDelay)
    
    

    add_button=driver.find_element(AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().description("Add")')
    add_button.click()
    print("Beneficiary added successfully.")
    exit_send_money(driver, timeout=timeout)

def send_money(driver,timeout=None):
    goto_bank(driver, timeout=timeout)
    scroll_down_recommended(driver)
    send_money_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Send Money'
    )
    send_money_button.click()
    #send_money_button.click()  # Click again to ensure the action is registered
    if timeout:
        time.sleep(timeout)
    print("Navigated to Send Money page.")
    
def exit_send_money(driver,timeout=None):
    back_button=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.Button").instance(0)'
    )
    back_button.click()
    print("Exited Send Money page.")
    if timeout:
        time.sleep(timeout)

def send_money_direct_pay(driver,upi_id="9444524432@yaml",amount="6969",timeout=None,typingDelay=0.1):
    send_money(driver, timeout=timeout)
    typingDelay = typingDelay if typingDelay is not None else 0.1
    wait=WebDriverWait(driver, timeout if timeout else 10)
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, "Direct Pay")))
    # Ensure the Direct Pay button is present before clicking
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
    slowTyping(upi_id_field, upi_id, delay=typingDelay)
    
    

    amount_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(1)'
    )
    amount_field.click()
    slowTyping(amount_field, amount, delay=typingDelay)
    
   

    pay_btn=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().description("Pay")'
    )
    pay_btn.click()
    print("Payment initiated successfully.")
    if timeout:
        time.sleep(timeout)
    exit_send_money(driver, timeout=timeout)

def direct_pay(driver,account_number="124385490367",beneficiary_name="Deez Nuts",Nick_name="earpods",timeout=None,typingDelay=0.1):
    goto_bank(driver, timeout=timeout)
    typingDelay = typingDelay if typingDelay is not None else 0.1
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
    slowTyping(account_number_field, account_number, delay=typingDelay)

    reenter_account_number_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(1)'
    )
    reenter_account_number_field.click()
    slowTyping(reenter_account_number_field, account_number, delay=typingDelay)
    

    beneficiary_name_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(2)'
    )
    beneficiary_name_field.click()
    slowTyping(beneficiary_name_field, beneficiary_name, delay=typingDelay)
   

    nick_name_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(3)'
    )
    nick_name_field.click()
    slowTyping(nick_name_field, Nick_name, delay=typingDelay)
    

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
    exit_send_money(driver, timeout=timeout)


def goto_all(driver,timeout=None):
    driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'All'
    ).click()
    if timeout:
        time.sleep(timeout)

def open_deposit(driver,full_name="Deez Nuts",mobile_number="1234567890",timeout=None,typingDelay=0.1):
    goto_all(driver,timeout=timeout)
    scroll_down_recommended(driver, distance=1400)
    typingDelay = typingDelay if typingDelay is not None else 0.1
    open_deposit_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Open Deposit'
    )
    open_deposit_button.click()
    print("Opening Deposit...")
    wait=WebDriverWait(driver, timeout if timeout else 10)
    wait.until(EC.presence_of_element_located((AppiumBy.ANDROID_UIAUTOMATOR, 'new UiSelector().className("android.widget.EditText").instance(0)')))
    full_name_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(0)'
    )
    full_name_field.click()
    slowTyping(full_name_field, full_name, delay=typingDelay)
  
    driver_hide_keyboard(driver, timeout=timeout)
    mobile_number_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(1)'
    )
    mobile_number_field.click()
    slowTyping(mobile_number_field, mobile_number, delay=typingDelay)
    

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
    
def goto_profile(driver,timeout=None):
    profile_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Profile'
    )
    profile_button.click()
    if timeout:
        time.sleep(timeout)
    
def change_mpin(driver,old_mpin="12345",new_mpin="54321",timeout=None,typingDelay=0.1):
    goto_profile(driver, timeout=timeout)
    wait=WebDriverWait(driver, timeout if timeout else 10)
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, 'Change MPIN')))
    print("Changing MPIN...")
    typingDelay = typingDelay if typingDelay is not None else 0.1
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
    slowTyping(old_mpin_field, old_mpin, delay=typingDelay)
    
    new_mpin_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(1)'
    )
    new_mpin_field.click()
    slowTyping(new_mpin_field, new_mpin, delay=typingDelay)
    

    change_Btn=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Cancel'
    )
    change_Btn.click()
    print("MPIN changed successfully.")
    if timeout:
        time.sleep(timeout)

def change_passcode(driver,old_passcode="12345",new_passcode="54321",timeout=None,typingDelay=0.1):
    goto_profile(driver, timeout=timeout)
    wait=WebDriverWait(driver, timeout if timeout else 10)
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, 'Change Passcode')))
    print("Changing Passcode...")   
    typingDelay = typingDelay if typingDelay is not None else 0.1
    change_passcode_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Change Passcode'
    )
    change_passcode_button.click()
    wait=WebDriverWait(driver, timeout if timeout else 10)
    wait.until(EC.presence_of_element_located((AppiumBy.ANDROID_UIAUTOMATOR, 'new UiSelector().className("android.widget.EditText").instance(0)')))
    old_passcode_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(0)'
    )
    old_passcode_field.click()
    slowTyping(old_passcode_field, old_passcode, delay=typingDelay)
    new_passcode_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(1)'
    )
    new_passcode_field.click()
    slowTyping(new_passcode_field, new_passcode, delay=typingDelay)
    change_Btn=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().description("Cancel")'
    )
    change_Btn.click()
    print("Passcode changed successfully.")
    if timeout:
        time.sleep(timeout)

def get_mmid(driver, timeout=None):
    goto_profile(driver, timeout=timeout)
    wait=WebDriverWait(driver, timeout if timeout else 10)
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, 'Get MMID')))
    mmid_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Get MMID'
    )
    mmid_button.click()
    if timeout:
        time.sleep(timeout)
    print("MNID retrieved successfully.")

    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, 'Copy')))
    copy_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Copy'
    )
    copy_button.click()
    print("MMID copied to clipboard.")
    cancell_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Close'
    )
    cancell_button.click()
    if timeout:
        time.sleep(timeout)

def view_my_account(driver, timeout=None):
    goto_profile(driver, timeout=timeout)
    wait=WebDriverWait(driver, timeout if timeout else 10)
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, 'My Accounts')))
    my_account_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'My Accounts'
    )
    my_account_button.click()
    if timeout:
        time.sleep(timeout)
    print("Viewed My Account section.")
    if timeout:
        time.sleep(timeout)
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, 'Back')))
    back_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Back'
    )
    back_button.click()
    print("Exited My Account section.")
    if timeout:
        time.sleep(timeout)

def toggle_upi_services(driver, enable=True, timeout=None):
    goto_profile(driver, timeout=timeout)
    scroll_down_recommended(driver, distance=1000)
    wait=WebDriverWait(driver, timeout if timeout else 10)
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, 'Block/Unblock UPI Services')))
    block_unblock_upi_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Block/Unblock UPI Services'
    )
    block_unblock_upi_button.click()
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, 'Cancel')))
    try:
        btn = driver.find_element(AppiumBy.ACCESSIBILITY_ID, "Block")
        print("Found 'Block' button.")
    except:
        btn = driver.find_element(AppiumBy.ACCESSIBILITY_ID, "Unblock")
        print("Found 'Unblock' button.")
    btn.click()
    print(f"UPI services {'enabled' if enable else 'disabled'} successfully.")
    if timeout:
        time.sleep(timeout)
    cancel_button = driver.find_element(AppiumBy.ACCESSIBILITY_ID, "Cancel")
    cancel_button.click()
    print("Cancelled UPI services toggle operation.")
    scroll_up_recommended(driver, distance=1000)
    

def goto_bank(driver,timeout=None):
    bank_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Bank'
    )
    bank_button.click()

    
    wait=WebDriverWait(driver, timeout if timeout else 10)
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID,'My Dashboard')))
    my_dashboard=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'My Dashboard'
    )
    my_dashboard.click()
    if timeout:
        time.sleep(timeout)
    print("Navigated to Bank section.")

def do_card_less_pay(driver,amount="6969",mobile_number="1123456789",timeout=None,typingDelay=0.1):
    goto_bank(driver, timeout=timeout)
    wait=WebDriverWait(driver, timeout if timeout else 10)
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, 'Card-less Cash')))
    print("Initiating Card-less Pay...")
    typingDelay = typingDelay if typingDelay is not None else 0.1
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
    slowTyping(amount_field, amount, delay=typingDelay)
    
    

    mobile_number_field=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.EditText").instance(1)'
    )   
    mobile_number_field.click()
    slowTyping(mobile_number_field, mobile_number, delay=typingDelay)
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

def view_transaction_history(driver,timeout=None):
    goto_bank(driver, timeout=timeout)
    wait=WebDriverWait(driver, timeout if timeout else 10)
    wait.until(EC.presence_of_element_located((AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().description("History")')))
    history_button=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().description("History")'
    )
    history_button.click()
    if timeout:
        time.sleep(2*timeout)
    print("Transaction history viewed successfully.")
    exit_transaction_history(driver, timeout=timeout)
def exit_transaction_history(driver,timeout=None):
    wait=WebDriverWait(driver, timeout if timeout else 10)
    wait.until(EC.presence_of_element_located((AppiumBy.ANDROID_UIAUTOMATOR, 'new UiSelector().className("android.widget.Button").instance(0)')))
    print("Exiting transaction history...")
    back_button=driver.find_element(
        AppiumBy.ANDROID_UIAUTOMATOR,'new UiSelector().className("android.widget.Button").instance(0)'
    )
    back_button.click()
    if timeout:
        time.sleep(timeout)
    print("Exited transaction history.")


def view_cards(driver,timeout=None):
    """"View debit card first then credit card."""
    cards_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Cards'
    )
    cards_button.click()
    if timeout:
        time.sleep(timeout)
    print("Entering into Cards section.")
    wait=WebDriverWait(driver, timeout if timeout else 10)
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, 'View Debit Cards')))
    view_debit_cards_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'View Debit Cards'
    )
    view_debit_cards_button.click()
    
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, 'Back')))
    print("Viewing Debit Cards.")
    back_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Back'
    )
    back_button.click()
    
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, 'View Credit Cards')))
    view_credit_cards_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'View Credit Cards'
    )
    view_credit_cards_button.click()
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, 'Back')))
    print("Viewing Credit Cards.")  
    back_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Back'
    )
    back_button.click()
    if timeout:
        time.sleep(timeout)
    print("Exited Cards section.")

def view_cards_reverse(driver,timeout=None):
    """View credit card first then debit card."""
    cards_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Cards'
    )
    cards_button.click()
    if timeout:
        time.sleep(timeout)
    print("Entering into Cards section.")
    wait=WebDriverWait(driver, timeout if timeout else 10)

    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, 'View Credit Cards')))
    view_credit_cards_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'View Credit Cards'
    )
    view_credit_cards_button.click()
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, 'Back')))
    print("Viewing Credit Cards.")  
    back_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Back'
    )
    back_button.click()

    
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, 'View Debit Cards')))
    view_debit_cards_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'View Debit Cards'
    )
    view_debit_cards_button.click()
    
    wait.until(EC.presence_of_element_located((AppiumBy.ACCESSIBILITY_ID, 'Back')))
    print("Viewing Debit Cards.")
    back_button=driver.find_element(
        AppiumBy.ACCESSIBILITY_ID,'Back'
    )
    back_button.click()
    
    
    if timeout:
        time.sleep(timeout)
    print("Exited Cards section.")


def get_random_delay(center=0.5, variation=0.2):
    """
    Returns a random delay centered around 'center' with +/- 'variation'.
    Default range: 0.3 to 0.7 seconds.
    """
    return random.uniform(center - variation, center + variation)