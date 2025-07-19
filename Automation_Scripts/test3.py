from components import initialize_driver,send_money_direct_pay,send_money, direct_pay, open_deposit, goto_bank, do_card_less_pay,exit_send_money
from components import add_benificiary,view_transaction_history,view_cards,view_cards_reverse
from swipe_down import scroll_down_recommended, scroll_up_recommended
from components import goto_profile,change_mpin, change_passcode,get_mmid,view_my_account,toggle_upi_services
driver = initialize_driver(timeout=2)
print("Starting main script...")
#goto_bank(driver, timeout=5)
#print("Sending money...")

#send_money_direct_pay(driver, upi_id="9444452444@yaml", amount="6969", timeout=3)
#exit_send_money(driver, timeout=5)
#add_benificiary(driver, name="Deez Nuts", upi_id="123456789@upi", mobile_number="1234567890", timeout=5)
#exit_send_money(driver, timeout=5)
#print("Direct pay...")
#direct_pay(driver, timeout=5)
#view_transaction_history(driver, timeout=5)
#view_cards(driver, timeout=5)
#view_cards_reverse(driver, timeout=5)
goto_profile(driver, timeout=5)
#scroll_down_recommended(driver, distance=1000)
toggle_upi_services(driver, timeout=5)
#scroll_up_recommended(driver, distance=600)

#change_mpin(driver, old_mpin="12345", new_mpin="54321", timeout=5)
#change_passcode(driver, old_passcode="12345", new_passcode="54321", timeout=5)
#get_mmid(driver, timeout=5)
#view_my_account(driver, timeout=5)