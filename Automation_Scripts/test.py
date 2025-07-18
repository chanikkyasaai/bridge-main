from components import initialize_driver,scan_biometric, secondary_login_button,send_money,send_money_direct_pay
from components import direct_pay,open_deposit,go_to_profile,change_mpin,goto_bank,do_card_less_pay

driver=initialize_driver(timeout=2)
##scan_biometric(driver, timeout=5)

#send_money_direct_pay(driver, upi_id="9444452444@yaml", amount="6969", timeout=3)
#direct_pay(driver)
send_money(driver, timeout=5)
#open_deposit(driver, timeout=5)
#goto_bank(driver, timeout=5)
#do_card_less_pay(driver, timeout=5)