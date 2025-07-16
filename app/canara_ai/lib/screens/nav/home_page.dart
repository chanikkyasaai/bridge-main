import 'package:canara_ai/logging/behaviour_route_tracker.dart';
import 'package:canara_ai/logging/button_wrapper.dart';
import 'package:canara_ai/logging/log_touch_data.dart';
import 'package:canara_ai/logging/logger_instance.dart';
import 'package:canara_ai/logging/monitor_logging.dart';
import 'package:canara_ai/main.dart';
import 'package:canara_ai/screens/nav/banking_page.dart';
import 'package:canara_ai/screens/nav/cards_page.dart';
import 'package:canara_ai/screens/nav/profile_page.dart';
import 'package:canara_ai/widgets/notification_sheet.dart';
import 'package:canara_ai/widgets/qr_scan_sheet.dart';
import 'package:canara_ai/widgets/search_sheet.dart';
import 'package:dio/dio.dart';
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
  bool _subscribed = false;

  final Color canaraBlue = const Color(0xFF0072BC);
  final Color canaraYellow = const Color(0xFFFFD600);
  final Color canaraLightBlue = const Color(0xFF00B9F1);
  final Color canaraDarkBlue = const Color(0xFF003366);
  final Color canaraPurple = const Color(0xFF7B1FA2);

  final dio = Dio();

  late final BehaviorLogger logger;
  late BehaviorRouteTracker tracker;

  Future<bool> _onWillPop(BuildContext context) async {
    final shouldExit = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Exit App'),
        content: Text('Do you want to close the app?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: Text('No'),
          ),
          TextButton(
            onPressed: () async {
              BehaviorMonitorState? monitorState = context.findAncestorStateOfType<BehaviorMonitorState>();
              await monitorState?.sendUserCloseEvent();

              print('end session'); 

              await logger.endSession('app_close');

              if(context.mounted) {
                Navigator.of(context).pop(true);
              }
            },
            child: Text('Yes'),
          ),
        ],
      ),
    );
    return shouldExit ?? false;
  }


  void _showSearchSheet() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => SearchSheet(
          canaraBlue: canaraBlue,
          canaraDarkBlue: canaraDarkBlue,
          logger: logger,
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

  @override
  void initState() {
    super.initState();
    logger = AppLogger.logger;
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_subscribed) {
      final route = ModalRoute.of(context);
      if (route is PageRoute) {
        tracker = BehaviorRouteTracker(logger, context);
        routeObserver.subscribe(tracker, route);
        _subscribed = true;
      }
    }
  }

  @override
  void dispose() {
    routeObserver.unsubscribe(tracker);
    super.dispose();
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
    return WillPopScope(
        onWillPop: () async => await _onWillPop(context),
        child: Scaffold(
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
            onPressed: () {
              logger.sendEvent('fab_qr_tap', {'from': _tabName(_selectedIndex)}); // ✅ log QR tap here too
              _showQRScanSheet();
            },
            child: const Icon(Icons.qr_code, color: Colors.white),
          ),
          bottomNavigationBar: _customNavBar(),
        ));
  }

  void _onNavBarItemTapped(int index, String label) {
    if (index == _selectedIndex) return;

    logger.sendEvent('nav_bar_tap', {
      'from': _selectedIndex,
      'to': index,
      'tab_name': label,
    });

    setState(() {
      _selectedIndex = index;
    });
  }

  String _tabName(int index) {
    switch (index) {
      case 0:
        return 'Home';
      case 1:
        return 'Banking';
      case 2:
        return 'History';
      case 3:
        return 'Cards';
      case 4:
        return 'Profile';
      default:
        return 'Unknown';
    }
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
                      LoggedButton(
                        eventName: 'button_press',
                        logger: logger,
                        eventData: {
                          'button_name': 'toggle_show_portfolio',
                          'new_state': !_showBalance,
                          'screen': 'Home Page',
                        },
                        onTap: () => setState(() => _showBalance = !_showBalance),
                        child: Row(
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
            _serviceItem('assets/icons/bank.png', 'Send Money', '/sendmoney'),
            _serviceItem('assets/icons/money.png', 'Direct Pay', '/directpay'),
            _serviceItem('assets/icons/mobilephone.png', 'My Beneficiary', '/addbeneficiary'),
            _serviceItem('assets/icons/passbook.png', 'ePassbook', '/epassbook'),
            _serviceItem('assets/icons/send-money.png', 'Card-less Cash', '/cardlesscash'), // Update if you have a route
            _serviceItem('assets/icons/contact-book.png', 'Donation', '/donation'),
            _serviceItem('assets/icons/folder.png', 'History', '/history'),
            _serviceItem('assets/icons/mobile-banking.png', 'Manage Accounts', '/manage-accounts'),
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
            _serviceItem('assets/icons/upi.svg', 'Register', '/registerupi'),
            _serviceItem('assets/icons/upi.svg', 'Scan any UPI QR', '/qrscan'),
            _serviceItem('assets/icons/bhim.png', 'Send Money to any UPI app', '/sendmoney'),
            _serviceItem('assets/icons/mobilephone.png', 'Pay to Contact/\nMobile Number', '/paycontact'),
            _serviceItem('assets/icons/cash.png', 'Approve Payment', '/approvepayment'),
            _serviceItem('assets/icons/creditrupee.png', 'Add RuPay Credit Card', '/addcreditcard'),
            _serviceItem('assets/icons/tap-to-pay.png', 'Tap & Pay', '/tappay'),
            _serviceItem('assets/icons/upi.svg', 'UPI Lite', '/upilite'),
          ]),

          _sectionTitle('Deposits'),
          _serviceGrid([
            _serviceItem('assets/icons/safety-box.png', 'Open Deposit', '/registerupi'),
            _serviceItem('assets/icons/file.png', 'Term Deposit Receipt', '/registerupi'),
            _serviceItem('assets/icons/add-document.png', 'Canara Dhanvarsha TD', '/registerupi'),
            _serviceItem('assets/icons/documents.png', 'RD Details', '/registerupi'),
            _serviceItem('assets/icons/paper.png', 'Payment of RD Installment', '/registerupi'),
            _serviceItem('assets/icons/invoice.png', 'Pre Mature Closure of RD/FD', '/registerupi'),
            _serviceItem('assets/icons/cancel.png', 'Close Fixed Deposit', '/registerupi'),
            _serviceItem('assets/icons/compose.png', 'Modify Fixed Deposit', '/registerupi'),
          ]),

          _sectionTitle('Loans'),
          _serviceGrid([
            _serviceItem('assets/icons/calendar.png', 'Instant Overdraft', '/registerupi'),
            _serviceItem('assets/icons/banking.png', 'Loan Details', '/registerupi'),
            _serviceItem('assets/icons/save.png', 'Loan Repayment', '/registerupi'),
            _serviceItem('assets/icons/Choker.png', 'Gold OD', '/registerupi'),
            _serviceItem('assets/icons/heart.png', 'Canara HEAL', '/registerupi'),
            _serviceItem('assets/icons/rupees.png', 'Loan Against Mutual Funds', '/registerupi'),
            _serviceItem('assets/icons/statement.png', 'Loan Account Statement', '/registerupi'),
            _serviceItem('assets/icons/tax.png', 'Actual Interest Collected', '/registerupi'),
          ]),

          _sectionTitle('LifeStyle'),
          _serviceGrid([
            _serviceItem('assets/icons/train.png', 'Train Tickets', '/registerupi'),
            _serviceItem('assets/icons/departures.png', 'Flights', '/registerupi'),
            _serviceItem('assets/icons/speedometer.png', 'Free Credit Score', '/registerupi'),
            _serviceItem('assets/icons/shopping-cart.png', 'Shopping', '/registerupi'),
            _serviceItem('assets/icons/mobile.png', 'Recharge', '/registerupi'),
            _serviceItem('assets/icons/pin.png', 'Experiences', '/registerupi'),
            _serviceItem('assets/icons/first-aid-kit.png', 'Healthcare', '/registerupi'),
            _serviceItem('assets/icons/card.png', 'E-Gift Card', '/registerupi'),
          ]),

          _sectionTitle('Stores & Offers'),
          _serviceGrid([
            _serviceItem('assets/icons/badge.png', 'Rewards', '/registerupi'),
            _serviceItem('assets/icons/fire.png', 'Hot Deals', '/registerupi'),
            _serviceItem('assets/icons/discount.png', 'Offers', '/registerupi'),
            _serviceItem('assets/icons/flipkart-icon.png', 'Flipkart', '/registerupi'),
            _serviceItem('assets/icons/amazon.svg', 'Amazon', '/registerupi'),
            _serviceItem('assets/icons/myntra.svg', 'Myntra', '/registerupi'),
            _serviceItem('assets/icons/amazon-prime-video.svg', 'Amazon Prime', '/registerupi'),
            _serviceItem('assets/icons/airtel.svg', 'Airtel Postpaid', '/registerupi'),
          ]),

          _sectionTitle('FOREX'),
          _serviceGrid([
            _serviceItem('assets/icons/money1.png', 'FOREX Beneficiary Mgmt', '/registerupi'),
            _serviceItem('assets/icons/exchange.png', 'Outward Remittance', '/registerupi'),
            _serviceItem('assets/icons/money-currency.png', 'Exchange Rate Enquiry', '/registerupi'),
            _serviceItem('assets/icons/trade.png', 'Inward Remittance', '/registerupi'),
          ]),

          _sectionTitle('Accounts & Services'),
          _serviceGrid([
            _serviceItem('assets/icons/safety-box.png', 'Apply for Locker', '/registerupi'),
            _serviceItem('assets/icons/analysis.png', 'Wealth Management', '/registerupi'),
            _serviceItem('assets/icons/filesearch.png', 'NACH Mandate Cancellation', '/registerupi'),
            _serviceItem('assets/icons/cheque.png', 'Cheque Book Request & Track', '/registerupi'),
            _serviceItem('assets/icons/credit-card.png', 'Cheque Status', '/registerupi'),
            _serviceItem('assets/icons/card-payment-cancel.png', 'Stop Cheque', '/registerupi'),
            _serviceItem('assets/icons/give.png', 'Positive Pay System', '/registerupi'),
            _serviceItem('assets/icons/candidacy.png', 'Nominee Maintenance', '/registerupi'),
          ]),

          _sectionTitle('GST & Other Taxes'),
          _serviceGrid([
            _serviceItem('assets/icons/money-1.png', 'Pay GST', '/registerupi'),
            _serviceItem('assets/icons/money-1.png', 'Check GST Payment Status', '/registerupi'),
            _serviceItem('assets/icons/money-1.png', 'Generate Receipt', '/registerupi'),
            _serviceItem('assets/icons/computer.png', 'Online Tax Payment', '/registerupi'),
          ]),

          _sectionTitle('Invest & Insure'),
          _serviceGrid([
            _serviceItem('assets/icons/life-insurance.png', 'Canara HSBC Life Insurance', '/registerupi'),
            _serviceItem('assets/icons/rupee-1.png', 'ASBA', '/registerupi'),
            _serviceItem('assets/icons/chart-file.png', 'Demat/Trade', '/registerupi'),
            _serviceItem('assets/icons/google-docs.png', '26AS', '/registerupi'),
            _serviceItem('assets/icons/rupees.png', 'Mutual Fund', '/registerupi'),
            _serviceItem('assets/icons/insurance.png', 'Home Insurance', '/registerupi'),
            _serviceItem('assets/icons/healthcare.png', 'Health Insurance', '/registerupi'),
            _serviceItem('assets/icons/protection.png', 'Motor Insurance', '/registerupi'),
          ]),

          _sectionTitle('Other Services'),
          _serviceGrid([
            _serviceItem('assets/icons/toll-road.png', 'Apply for FASTag', '/registerupi'),
            _serviceItem('assets/icons/toll-road.png', 'Manage FASTag', '/registerupi'),
            _serviceItem('assets/icons/box.png', 'Donate to PM Cares', '/registerupi'),
            _serviceItem('assets/icons/calendar.png', 'Calendar', '/registerupi'),
            _serviceItem('assets/icons/aging.png', 'Pension Seva Portal', '/registerupi'),
            _serviceItem('assets/icons/service.png', 'Service Charges', '/registerupi'),
            _serviceItem('assets/icons/lock.png', 'Block/Unblock IB', '/registerupi'),
            _serviceItem('assets/icons/rotation-lock.png', 'Reset IB Login Password', '/registerupi'),
          ]),

          _sectionTitle('Kisan Services'),
          _serviceGrid([
            _serviceItem('assets/icons/balance.png', 'Mandi Prices', '/registerupi'),
            _serviceItem('assets/icons/cloud.png', 'Weather Update', '/registerupi'),
            _serviceItem('assets/icons/market.png', 'Market Place Integration', '/registerupi'),
            _serviceItem('assets/icons/crop.png', 'Crop Advisory & Predictive Alerts', '/registerupi'),
            _serviceItem('assets/icons/supply-chain.png', 'Value Chain Finance', '/registerupi'),
            _serviceItem('assets/icons/warehouse.png', 'Warehouse Receipt Finance', '/registerupi'),
            _serviceItem('assets/icons/calendar.png', 'Short Term Loans For Farmers', '/registerupi'),
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

  Widget _serviceItem(String asset, String label, String routeName) {
    return GestureDetector(
      onTap: () {
        Navigator.of(context).pushNamed(routeName);
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
              child: (asset.endsWith("svg"))
                  ? SvgPicture.asset(asset, height: 26)
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
      onTap: () => _onNavBarItemTapped(index, label),
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
