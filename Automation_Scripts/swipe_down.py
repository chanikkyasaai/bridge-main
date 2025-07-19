from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.pointer_input import PointerInput
from selenium.webdriver.common.actions import interaction
from selenium.webdriver.common.actions import wheel_input
from selenium.webdriver.common.actions.action_builder import ActionBuilder
import time

# Define scroll gesture using W3C Actions
def scroll_down(driver, distance=300, duration=500):
    size = driver.get_window_size()
    start_x = size['width'] // 2
    start_y = size['height'] // 2
    end_y = start_y - distance  # swipe up to scroll down
    
    actions = ActionBuilder(driver)
    finger = PointerInput(interaction.POINTER_TOUCH, "finger")
    
    # Add actions to the builder (not key_action)
    actions.add_action(finger.create_pointer_move(0, PointerInput.Origin.VIEWPORT, start_x, start_y))
    actions.add_action(finger.create_pointer_down(PointerInput.PointerEventProperties.BUTTON_LEFT))
    actions.add_action(finger.create_pointer_move(duration, PointerInput.Origin.VIEWPORT, start_x, end_y))
    actions.add_action(finger.create_pointer_up(PointerInput.PointerEventProperties.BUTTON_LEFT))
    
    actions.perform()

def scroll_up(driver, distance=300, duration=500):
    size = driver.get_window_size()
    start_x = size['width'] // 2
    start_y = size['height'] // 2
    end_y = start_y + distance  # swipe down to scroll up
    
    actions = ActionBuilder(driver)
    finger = PointerInput(interaction.POINTER_TOUCH, "finger")
    
    actions.add_action(finger.create_pointer_move(0, PointerInput.Origin.VIEWPORT, start_x, start_y))
    actions.add_action(finger.create_pointer_down(PointerInput.PointerEventProperties.BUTTON_LEFT))
    actions.add_action(finger.create_pointer_move(duration, PointerInput.Origin.VIEWPORT, start_x, end_y))
    actions.add_action(finger.create_pointer_up(PointerInput.PointerEventProperties.BUTTON_LEFT))
    
    actions.perform()
# Alternative simpler approach using ActionChains
def scroll_down_simple(driver, distance=300, duration=500):
    size = driver.get_window_size()
    start_x = size['width'] // 2
    start_y = size['height'] // 2
    end_y = start_y - distance
    
    actions = ActionChains(driver)
    finger = PointerInput(interaction.POINTER_TOUCH, "finger")
    
    actions.w3c_actions = ActionBuilder(driver, mouse=finger)
    actions.w3c_actions.pointer_action.move_to_location(start_x, start_y)
    actions.w3c_actions.pointer_action.pointer_down()
    actions.w3c_actions.pointer_action.move_to_location(start_x, end_y)
    actions.w3c_actions.pointer_action.pointer_up()
    
    actions.perform()

def scroll_up_simple(driver, distance=300, duration=500):
    size = driver.get_window_size()
    start_x = size['width'] // 2
    start_y = size['height'] // 2
    end_y = start_y + distance
    
    actions = ActionChains(driver)
    finger = PointerInput(interaction.POINTER_TOUCH, "finger")
    
    actions.w3c_actions = ActionBuilder(driver, mouse=finger)
    actions.w3c_actions.pointer_action.move_to_location(start_x, start_y)
    actions.w3c_actions.pointer_action.pointer_down()
    actions.w3c_actions.pointer_action.move_to_location(start_x, end_y)
    actions.w3c_actions.pointer_action.pointer_up()
    
    actions.perform()

# Most recommended approach for mobile scrolling
def scroll_down_recommended(driver, distance=300):
    size = driver.get_window_size()
    start_x = size['width'] // 2
    start_y = size['height'] // 2
    end_y = start_y - distance
    
    # Using driver.swipe (if available) - most reliable for mobile
    try:
        driver.swipe(start_x, start_y, start_x, end_y, duration=500)
    except AttributeError:
        # Fallback to ActionChains if swipe is not available
        actions = ActionChains(driver)
        actions.move_to_location(start_x, start_y)
        actions.click_and_hold()
        actions.move_to_location(start_x, end_y)
        actions.release()
        actions.perform()
def scroll_up_recommended(driver, distance=300):
    size = driver.get_window_size()
    start_x = size['width'] // 2
    start_y = size['height'] // 2
    end_y = start_y + distance
    
    # Using driver.swipe (if available) - most reliable for mobile
    try:
        driver.swipe(start_x, start_y, start_x, end_y, duration=500)
    except AttributeError:
        # Fallback to ActionChains if swipe is not available
        actions = ActionChains(driver)
        actions.move_to_location(start_x, start_y)
        actions.click_and_hold()
        actions.move_to_location(start_x, end_y)
        actions.release()
        actions.perform()