from components import initialize_driver,driver_hide_keyboard,go_to_registration_page,fill_registration_form,close_driver

import time

driver= initialize_driver(timeout=5)




print("Starting login script...")
print("Going to registration page...")

go_to_registration_page(driver, timeout=2)



print("Filling registration form...")
fill_registration_form(driver, phone_number="9444452444", password="123456", confirm_password="12345", timeout=2)
close_driver(driver)