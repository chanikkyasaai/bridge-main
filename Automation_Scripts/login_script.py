from components import initialize_driver,login_phone_number_element, driver_hide_keyboard,login_password_element,login_button_click
import time



##Creating a driver instance with a timeout
driver = initialize_driver(timeout=10)



print("Starting Login script...")
print("Going to login page...")
time.sleep(5)
print("Entering Phone number ....")


login_phone_number_element(driver, phone_number="9444452444", timeout=2)

driver_hide_keyboard(driver, timeout=2)

print("Entering Password....")
login_password_element(driver, password="12345", timeout=2)


login_button_click(driver)