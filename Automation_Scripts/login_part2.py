import time
from components import initialize_driver, login_phone_number_element, driver_hide_keyboard, login_password_element, login_button_click, secondary_login_button
from components import scan_biometric
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy

import asyncio


from components import send_money
driver=initialize_driver(timeout=2)
async def main():
    send_money_btn = driver.find_element(AppiumBy.ACCESSIBILITY_ID, "Direct Pay")
    print(send_money_btn.is_displayed())
    send_money_btn.click()
    send_money_btn.click()

asyncio.run(main())


#send_money(driver,timeout=5)