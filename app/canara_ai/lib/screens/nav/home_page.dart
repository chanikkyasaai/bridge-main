import 'package:canara_ai/routes/slide_route.dart';
import 'package:canara_ai/screens/nav/banking_page.dart';
import 'package:canara_ai/screens/nav/cards_page.dart';
import 'package:canara_ai/screens/nav/profile/manage_view_balance_page.dart';
import 'package:canara_ai/screens/nav/profile/upi_lite_page.dart';
import 'package:canara_ai/screens/nav/profile_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/cardless_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/direct_pay_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/donation_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/epassbook_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/history_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/my_beneficiary_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/send_money_page.dart';
import 'package:canara_ai/screens/nav/tabs/upi/add_credit_card.dart';
import 'package:canara_ai/screens/nav/tabs/upi/approve_payment.dart';
import 'package:canara_ai/screens/nav/tabs/upi/pay_contact.dart';
import 'package:canara_ai/screens/nav/tabs/upi/register_upi.dart';
import 'package:canara_ai/screens/nav/tabs/upi/scan_qr.dart';
import 'package:canara_ai/screens/nav/tabs/upi/send_money_upi.dart';
import 'package:canara_ai/screens/nav/tabs/upi/tap_pay.dart';
import 'package:canara_ai/widgets/notification_sheet.dart';
import 'package:canara_ai/widgets/qr_scan_sheet.dart';
import 'package:canara_ai/widgets/search_sheet.dart';
import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _selectedIndex = 0;
  bool _showBalance = true;

  final Color canaraBlue = const Color(0xFF0072BC);
  final Color canaraYellow = const Color(0xFFFFD600);
  final Color canaraLightBlue = const Color(0xFF00B9F1);
  final Color canaraDarkBlue = const Color(0xFF003366);
  final Color canaraPurple = const Color(0xFF7B1FA2);

  void _showSearchSheet() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => SearchSheet(
          canaraBlue: canaraBlue,
          canaraDarkBlue: canaraDarkBlue,
        ),
      ),
    );
  }

  void _showNotificationSheet() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
      ),
      builder: (context) {
        return NotificationSheet(canaraBlue: canaraBlue);
      },
    );
  }

  void _showQRScanSheet() async {
    await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.black.withOpacity(0.95),
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
      ),
      builder: (context) {
        return QRScanSheet(
          canaraBlue: canaraBlue,
        );
      },
    );
  }

  List<Widget> get _pages => [
        _mainHomePage(),
        const BankingPage(),
        const _DummyPage(label: 'History'), // Placeholder for History
        const CardsPage(),
        const ProfilePage(),
      ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF7F9FB),
      body: SafeArea(
        child: IndexedStack(
          index: _selectedIndex,
          children: _pages,
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
      floatingActionButton: FloatingActionButton(
        backgroundColor: canaraBlue,
        onPressed: _showQRScanSheet,
        child: const Icon(Icons.qr_code, color: Colors.white),
      ),
      bottomNavigationBar: _customNavBar(),
    );
  }

  Widget _mainHomePage() {
    return SingleChildScrollView(
      child: Column(
        children: [
          // Top Bar
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
            child: Row(
              children: [
                CircleAvatar(
                  backgroundImage: AssetImage('assets/icons/user.png'),
                  radius: 20,
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    'Logarathan',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                      color: canaraDarkBlue,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                Icon(Icons.currency_rupee, color: canaraBlue, size: 24),
                const SizedBox(width: 10),
                GestureDetector(
                  onTap: _showNotificationSheet,
                  child: Icon(Icons.notifications_none, color: canaraBlue, size: 24),
                ),
                const SizedBox(width: 10),
                GestureDetector(
                  onTap: _showSearchSheet,
                  child: Icon(Icons.search, color: canaraBlue, size: 24),
                ),
                const SizedBox(width: 10),
                Icon(Icons.help_outline, color: canaraBlue, size: 24),
              ],
            ),
          ),
          const SizedBox(height: 12),
          // Portfolio Card
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Container(
              width: double.infinity,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [canaraBlue, canaraLightBlue],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(18),
                boxShadow: [
                  BoxShadow(
                    color: canaraBlue.withOpacity(0.08),
                    blurRadius: 12,
                    offset: const Offset(0, 6),
                  ),
                ],
              ),
              padding: const EdgeInsets.all(18),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Top Row: Title + Logos
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.center,
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      Expanded(
                        child: Column(
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  'My Portfolio',
                                  style: TextStyle(
                                    fontSize: 18,
                                    color: Colors.white,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                Image.asset(
                                  'assets/images/transapplogo.png',
                                  height: 60,
                                  width: 60,
                                  fit: BoxFit.contain,
                                ),
                                Image.asset(
                                  'assets/images/transbiglogo.png',
                                  height: 50,
                                  fit: BoxFit.contain,
                                ),
                              ],
                            ),
                          ],
                        ),
                      )
                    ],
                  ),
                  const SizedBox(height: 16),
                  // Portfolio Grid with Amounts
                  Row(
                    children: [
                      // Left Column
                      Expanded(
                        child: Column(
                          children: [
                            _portfolioGridTileWithAmount(
                              icon: 'assets/icons/rupee.png',
                              label: 'Savings (A/C)',
                              color: Colors.white,
                              accent: Colors.transparent,
                              amount: 125430.75,
                            ),
                            const SizedBox(height: 18),
                            _portfolioGridTileWithAmount(
                              icon: 'assets/icons/donation.png',
                              label: 'Deposits (A/C)',
                              color: Colors.white,
                              accent: Colors.transparent,
                              amount: 100000.00,
                            ),
                          ],
                        ),
                      ),
                      // Right Column
                      Expanded(
                        child: Column(
                          children: [
                            _portfolioGridTileWithAmount(
                              icon: 'assets/icons/save.png',
                              label: 'OD Account (A/C)',
                              color: Colors.white,
                              accent: Colors.transparent,
                              amount: 23450.00,
                            ),
                            const SizedBox(height: 18),
                            _portfolioGridTileWithAmount(
                              icon: 'assets/icons/loan.png',
                              label: 'Loans (A/C)',
                              color: Colors.white,
                              accent: Colors.transparent,
                              amount: 54321.99,
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 18),
                  // Show/Hide Switch (optional, can be removed if not needed)
                  Row(
                    children: [
                      Switch(
                        value: _showBalance,
                        onChanged: (val) => setState(() => _showBalance = val),
                        activeColor: canaraYellow,
                        inactiveThumbColor: Colors.white,
                        inactiveTrackColor: Colors.white54,
                      ),
                      Text(
                        'Show',
                        style: TextStyle(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          // Banner
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0),
            child: Container(
              width: double.infinity,
              decoration: BoxDecoration(
                color: canaraYellow.withOpacity(0.18),
                borderRadius: BorderRadius.circular(12),
              ),
              padding: const EdgeInsets.all(12),
              child: Row(
                children: [
                  Image.asset('assets/images/medical.jpeg', height: 40),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      'Concession on MEDICAL EXPENSES upto 25%\nCanara JEEVAN DHARA',
                      style: TextStyle(color: canaraBlue, fontWeight: FontWeight.bold),
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 18),
          // Pay & Transfer Section
          _sectionTitle('Pay & Transfer'),
          _serviceGrid([
            _serviceItem('assets/icons/bank.png', 'Send Money', SendMoneyPage()),
            _serviceItem('assets/icons/money.png', 'Direct Pay', const DirectPayPage()),
            _serviceItem('assets/icons/mobilephone.png', 'My Beneficiary', MyBeneficiaryPage()),
            _serviceItem('assets/icons/passbook.png', 'ePassbook', const EPassbookPage()),
            _serviceItem('assets/icons/send-money.png', 'Card-less Cash', const CardlessCashPage()),
            _serviceItem('assets/icons/contact-book.png', 'Donation', const DonationPage()),
            _serviceItem('assets/icons/folder.png', 'History', const HistoryPage()),
            _serviceItem('assets/icons/mobile-banking.png', 'Manage Accounts', const ManageViewBalancePage()),
          ]),
          // UPI ID
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 10),
            child: Container(
              width: double.infinity,
              decoration: BoxDecoration(
                color: canaraBlue.withOpacity(0.08),
                borderRadius: BorderRadius.circular(10),
              ),
              padding: const EdgeInsets.all(10),
              child: Row(
                children: [
                  Icon(Icons.qr_code, color: canaraBlue),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text('UPI (My UPI ID: 70XXXXXXX78 @cnrb)', style: TextStyle(color: canaraBlue)),
                  ),
                  Text('More', style: TextStyle(color: canaraBlue, fontWeight: FontWeight.bold)),
                ],
              ),
            ),
          ),

          // --- INSERTED SECTIONS START HERE ---

          _sectionTitle('UPI'),
          _serviceGrid([
            _serviceItem('assets/icons/upi.svg', 'Register', RegisterPage()),
            _serviceItem('assets/icons/upi.svg', 'Scan any UPI QR', QRScanPage()),
            _serviceItem('assets/icons/bhim.png', 'Send Money to any UPI app', SendMoneyPageUPI()),
            _serviceItem('assets/icons/mobilephone.png', 'Pay to Contact/\nMobile Number', PayContactPage()),
            _serviceItem('assets/icons/cash.png', 'Approve Payment', ApprovePaymentPage()),
            _serviceItem('assets/icons/creditrupee.png', 'Add RuPay Credit Card', AddCreditCardPage()),
            _serviceItem('assets/icons/tap-to-pay.png', 'Tap & Pay', TapPayPage()),
            _serviceItem('assets/icons/upi.svg', 'UPI Lite', UpiLitePage()),
          ]),

          _sectionTitle('Deposits'),
          _serviceGrid([
            _serviceItem('assets/icons/safety-box.png', 'Open Deposit', RegisterPage()),
            _serviceItem('assets/icons/file.png', 'Term Deposit Receipt', RegisterPage()),
            _serviceItem('assets/icons/add-document.png', 'Canara Dhanvarsha TD', RegisterPage()),
            _serviceItem('assets/icons/documents.png', 'RD Details', RegisterPage()),
            _serviceItem('assets/icons/paper.png', 'Payment of RD Installment', RegisterPage()),
            _serviceItem('assets/icons/invoice.png', 'Pre Mature Closure of RD/FD', RegisterPage()),
            _serviceItem('assets/icons/cancel.png', 'Close Fixed Deposit', RegisterPage()),
            _serviceItem('assets/icons/compose.png', 'Modify Fixed Deposit', RegisterPage()),
          ]),

          _sectionTitle('Loans'),
          _serviceGrid([
            _serviceItem('assets/icons/calendar.png', 'Instant Overdraft', RegisterPage()),
            _serviceItem('assets/icons/banking.png', 'Loan Details', RegisterPage()),
            _serviceItem('assets/icons/save.png', 'Loan Repayment', RegisterPage()),
            _serviceItem('assets/icons/Choker.png', 'Gold OD', RegisterPage()),
            _serviceItem('assets/icons/heart.png', 'Canara HEAL', RegisterPage()),
            _serviceItem('assets/icons/rupees.png', 'Loan Against Mutual Funds', RegisterPage()),
            _serviceItem('assets/icons/statement.png', 'Loan Account Statement', RegisterPage()),
            _serviceItem('assets/icons/tax.png', 'Actual Interest Collected', RegisterPage()),
          ]),

          _sectionTitle('LifeStyle'),
          _serviceGrid([
            _serviceItem('assets/icons/train.png', 'Train Tickets', RegisterPage()),
            _serviceItem('assets/icons/departures.png', 'Flights', RegisterPage()),
            _serviceItem('assets/icons/speedometer.png', 'Free Credit Score', RegisterPage()),
            _serviceItem('assets/icons/shopping-cart.png', 'Shopping', RegisterPage()),
            _serviceItem('assets/icons/mobile.png', 'Recharge', RegisterPage()),
            _serviceItem('assets/icons/pin.png', 'Experiences', RegisterPage()),
            _serviceItem('assets/icons/first-aid-kit.png', 'Healthcare', RegisterPage()),
            _serviceItem('assets/icons/card.png', 'E-Gift Card', RegisterPage()),
          ]),

          _sectionTitle('Stores & Offers'),
          _serviceGrid([
            _serviceItem('assets/icons/badge.png', 'Rewards', RegisterPage()),
            _serviceItem('assets/icons/fire.png', 'Hot Deals', RegisterPage()),
            _serviceItem('assets/icons/discount.png', 'Offers', RegisterPage()),
            _serviceItem('assets/icons/flipkart-icon.png', 'Flipkart', RegisterPage()),
            _serviceItem('assets/icons/amazon.svg', 'Amazon', RegisterPage()),
            _serviceItem('assets/icons/myntra.svg', 'Myntra', RegisterPage()),
            _serviceItem('assets/icons/amazon-prime-video.svg', 'Amazon Prime', RegisterPage()),
            _serviceItem('assets/icons/airtel.svg', 'Airtel Postpaid', RegisterPage()),
          ]),

          _sectionTitle('FOREX'),
          _serviceGrid([
            _serviceItem('assets/icons/money1.png', 'FOREX Beneficiary Mgmt', RegisterPage()),
            _serviceItem('assets/icons/exchange.png', 'Outward Remittance', RegisterPage()),
            _serviceItem('assets/icons/money-currency.png', 'Exchange Rate Enquiry', RegisterPage()),
            _serviceItem('assets/icons/trade.png', 'Inward Remittance', RegisterPage()),
          ]),

          _sectionTitle('Accounts & Services'),
          _serviceGrid([
            _serviceItem('assets/icons/safety-box.png', 'Apply for Locker', RegisterPage()),
            _serviceItem('assets/icons/analysis.png', 'Wealth Management', RegisterPage()),
            _serviceItem('assets/icons/filesearch.png', 'NACH Mandate Cancellation', RegisterPage()),
            _serviceItem('assets/icons/cheque.png', 'Cheque Book Request & Track', RegisterPage()),
            _serviceItem('assets/icons/credit-card.png', 'Cheque Status', RegisterPage()),
            _serviceItem('assets/icons/card-payment-cancel.png', 'Stop Cheque', RegisterPage()),
            _serviceItem('assets/icons/give.png', 'Positive Pay System', RegisterPage()),
            _serviceItem('assets/icons/candidacy.png', 'Nominee Maintenance', RegisterPage()),
          ]),

          _sectionTitle('GST & Other Taxes'),
          _serviceGrid([
            _serviceItem('assets/icons/money-1.png', 'Pay GST', RegisterPage()),
            _serviceItem('assets/icons/money-1.png', 'Check GST Payment Status', RegisterPage()),
            _serviceItem('assets/icons/money-1.png', 'Generate Receipt', RegisterPage()),
            _serviceItem('assets/icons/computer.png', 'Online Tax Payment', RegisterPage()),
          ]),
          // Invest & Insure
          _sectionTitle('Invest & Insure'),
          _serviceGrid([
            _serviceItem('assets/icons/life-insurance.png', 'Canara HSBC Life Insurance', RegisterPage()),
            _serviceItem('assets/icons/rupee-1.png', 'ASBA', RegisterPage()),
            _serviceItem('assets/icons/chart-file.png', 'Demat/Trade', RegisterPage()),
            _serviceItem('assets/icons/google-docs.png', '26AS', RegisterPage()),
            _serviceItem('assets/icons/rupees.png', 'Mutual Fund', RegisterPage()),
            _serviceItem('assets/icons/insurance.png', 'Home Insurance', RegisterPage()),
            _serviceItem('assets/icons/healthcare.png', 'Health Insurance', RegisterPage()),
            _serviceItem('assets/icons/protection.png', 'Motor Insurance', RegisterPage()),
          ]),

          // REM Sections
          _sectionTitle('Other Services'),
          _serviceGrid([
            _serviceItem('assets/icons/toll-road.png', 'Apply for FASTag', RegisterPage()),
            _serviceItem('assets/icons/toll-road.png', 'Manage FASTag', RegisterPage()),
            _serviceItem('assets/icons/box.png', 'Donate to PM Cares', RegisterPage()),
            _serviceItem('assets/icons/calendar.png', 'Calendar', RegisterPage()),
            _serviceItem('assets/icons/aging.png', 'Pension Seva Portal', RegisterPage()),
            _serviceItem('assets/icons/service.png', 'Service Charges', RegisterPage()),
            _serviceItem('assets/icons/lock.png', 'Block/Unblock IB', RegisterPage()),
            _serviceItem('assets/icons/rotation-lock.png', 'Reset IB Login Password', RegisterPage()),
          ]),

          _sectionTitle('Kisan Services'),
          _serviceGrid([
            _serviceItem('assets/icons/balance.png', 'Mandi Prices', RegisterPage()),
            _serviceItem('assets/icons/cloud.png', 'Weather Update', RegisterPage()),
            _serviceItem('assets/icons/market.png', 'Market Place Integration', RegisterPage()),
            _serviceItem('assets/icons/crop.png', 'Crop Advisory & Predictive Alerts', RegisterPage()),
            _serviceItem('assets/icons/supply-chain.png', 'Value Chain Finance', RegisterPage()),
            _serviceItem('assets/icons/warehouse.png', 'Warehouse Receipt Finance', RegisterPage()),
            _serviceItem('assets/icons/calendar.png', 'Short Term Loans For Farmers', RegisterPage()),
          ]),

          // --- INSERTED SECTIONS END HERE ---

          // GST & Other Taxes (repeat, remove if not needed)

          const SizedBox(height: 24),
        ],
      ),
    );
  }

  Widget _sectionTitle(String title) {
    return Padding(
      padding: const EdgeInsets.only(left: 18, top: 10, bottom: 6),
      child: Align(
        alignment: Alignment.centerLeft,
        child: Text(
          title,
          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: canaraDarkBlue),
        ),
      ),
    );
  }

  Widget _serviceGrid(List<Widget> items) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16.0),
      child: GridView.count(
        crossAxisCount: 4,
        shrinkWrap: true,
        physics: const NeverScrollableScrollPhysics(),
        mainAxisSpacing: 14,
        crossAxisSpacing: 8,
        childAspectRatio: 0.85,
        children: items,
      ),
    );
  }

  Widget _serviceItem(String asset, String label, Widget destinationPage) {
    return GestureDetector(
      onTap: () {
        Navigator.of(context).push(SlideRightRoute(page: destinationPage));
      },
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(12),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.04),
                  blurRadius: 4,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: Center(
              child: (asset.endsWith("svg")) ?
                SvgPicture.asset(asset, height: 26)
              : Image.asset(
                asset,
                height: 26,
              ),
            ),
          ),
          const SizedBox(height: 6),
          Text(
            label,
            style: const TextStyle(
              fontSize: 11,
              color: Color(0xFF003366),
              fontWeight: FontWeight.w500,
            ),
            textAlign: TextAlign.center,
            maxLines: 2,
          ),
        ],
      ),
    );

  }

  Widget _customNavBar() {
    return BottomAppBar(
      shape: const CircularNotchedRectangle(),
      notchMargin: 8,
      color: Colors.white,
      elevation: 10,
      child: SizedBox(
        height: 62,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceAround,
          children: [
            _navBarItem(Icons.apps, 'All', 0),
            _navBarItem(Icons.account_balance, 'Bank', 1),
            const SizedBox(width: 48), // space for FAB
            _navBarItem(Icons.credit_card, 'Cards', 3),
            _navBarItem(Icons.person, 'Profile', 4),
          ],
        ),
      ),
    );
  }

  Widget _navBarItem(IconData icon, String label, int index) {
    final bool selected = _selectedIndex == index;
    return GestureDetector(
      onTap: () => setState(() => _selectedIndex = index),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, color: selected ? canaraBlue : Colors.grey, size: 26),
          const SizedBox(height: 2),
          Text(
            label,
            style: TextStyle(
              color: selected ? canaraBlue : Colors.grey,
              fontWeight: selected ? FontWeight.bold : FontWeight.normal,
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }

  Widget _portfolioGridTileWithAmount({
    required String icon,
    required String label,
    required Color color,
    required Color accent,
    required double amount,
  }) {
    return Row(
      children: [
        Container(
          width: 32,
          height: 32,
          decoration: BoxDecoration(
            color: accent,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Center(
            child: Image.asset(
              icon,
              height: 25,
              color: color,
              fit: BoxFit.contain,
            ),
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                label,
                style: TextStyle(
                  color: color,
                  fontWeight: FontWeight.bold,
                  fontSize: 12,
                ),
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
              ),
              const SizedBox(height: 2),
              Text(
                _showBalance ? '₹ ${amount.toStringAsFixed(2)}' : '₹ ••••••••',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 15,
                  letterSpacing: 1,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

// Dummy page for History tab
class _DummyPage extends StatelessWidget {
  final String label;
  const _DummyPage({required this.label});
  @override
  Widget build(BuildContext context) {
    return Center(child: Text(label, style: const TextStyle(fontSize: 20)));
  }
}
