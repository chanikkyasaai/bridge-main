import 'package:canara_ai/logging/behaviour_route_tracker.dart';
import 'package:canara_ai/logging/button_wrapper.dart';
import 'package:canara_ai/logging/log_touch_data.dart';
import 'package:canara_ai/logging/logger_instance.dart';
import 'package:canara_ai/main.dart';
import 'package:canara_ai/screens/nav/tabs/banking/manage_accounts_page.dart';
import 'package:canara_ai/screens/nav/tabs/banking/statement_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/add_beneficiary_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/cardless_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/direct_pay_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/epassbook_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/history_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/my_beneficiary_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/send_money_page.dart';
import 'package:flutter/material.dart';

class BankingPage extends StatefulWidget {
  const BankingPage({super.key});

  @override
  State<BankingPage> createState() => _BankingPageState();
}

class _BankingPageState extends State<BankingPage> {
  int _selectedTab = 1; // 0: My Beneficiary, 1: My Dashboard, 2: Frequently Used
  bool _isBalanceVisible = false;
  final String _accountBalance = "₹1,25,480.50";

  late BehaviorLogger logger;
  late BehaviorRouteTracker tracker;
  bool _subscribed = false;

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
  void initState() {
    super.initState();
    logger = AppLogger.logger;
  }

  @override
  void dispose() {
    routeObserver.unsubscribe(tracker);
  }

  @override
  Widget build(BuildContext context) {
    final Color canaraBlue = const Color(0xFF0072BC);
    final Color canaraLightBlue = const Color(0xFF00B9F1);
    final Color canaraPurple = const Color(0xFF7B1FA2);
    final Color canaraYellow = const Color(0xFFFFD600);
    final Color canaraDarkBlue = const Color(0xFF003366);

    return Scaffold(
      backgroundColor: const Color(0xFFF7F9FB),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: const Text(
          'My Banking',
          style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
        iconTheme: IconThemeData(color: canaraBlue),
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            // Account Card
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 18.0, vertical: 8),
              child: Container(
                width: double.infinity,
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [canaraBlue, canaraLightBlue],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(18),
                ),
                padding: const EdgeInsets.all(18),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Container(
                          width: 34,
                          height: 34,
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: Icon(Icons.account_balance, color: canaraBlue),
                        ),
                        const SizedBox(width: 10),
                        Text(
                          'Savings',
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                            fontSize: 16,
                          ),
                        ),
                        const Spacer(),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: canaraPurple,
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: const Text(
                            '110084150765',
                            style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 13),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    Row(
                      children: [
                        LoggedButton(
                          logger: logger,
                          eventName: 'button_press',
                          eventData: {
                            'button_name': 'view_balance',
                            'new_state': !_isBalanceVisible,
                            'screen': 'Banking Page',
                          },
                          onTap: () {
                            setState(() {
                              _isBalanceVisible = !_isBalanceVisible;
                            });
                          },
                          onDoubleTap: () {},
                          onLongPress: () {},
                          child: Row(
                            children: [
                              Icon(
                                _isBalanceVisible ? Icons.visibility : Icons.visibility_off,
                                color: Colors.white,
                                size: 22,
                              ),
                              const SizedBox(width: 8),
                              Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    _isBalanceVisible ? 'Available Balance' : 'View Balance',
                                    style: TextStyle(
                                      color: Colors.white.withOpacity(0.9),
                                      fontSize: 12,
                                    ),
                                  ),
                                  if (_isBalanceVisible)
                                    Text(
                                      _accountBalance,
                                      style: TextStyle(
                                        color: Colors.white,
                                        fontWeight: FontWeight.bold,
                                        fontSize: 18,
                                      ),
                                    ),
                                ],
                              ),
                            ],
                          ),
                        ),
                        const Spacer(),
                        TextButton(
                          onPressed: () {
                            Navigator.push(
                              context,
                              MaterialPageRoute(builder: (context) => const StatementPage()),
                            );
                          },
                          child: const Text('Statement', style: TextStyle(color: Colors.white)),
                        ),
                        TextButton(
                          onPressed: () {
                            Navigator.push(
                              context,
                              MaterialPageRoute(builder: (context) => const ManageAccountPage()),
                            );
                          },
                          child: const Text('Manage', style: TextStyle(color: Colors.white)),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            // Tabs
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 18.0, vertical: 6),
              child: Row(
                children: [
                  _tabButton('My Beneficiary', _selectedTab == 0, canaraBlue, 0),
                  _tabButton('My Dashboard', _selectedTab == 1, canaraBlue, 1),
                  _tabButton('Frequently Used', _selectedTab == 2, canaraBlue, 2),
                ],
              ),
            ),
            // Tab Content
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 18.0, vertical: 8),
              child: _getTabContent(canaraBlue, canaraYellow, canaraLightBlue, canaraPurple),
            ),
            // Banner
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 18.0, vertical: 10),
              child: Container(
                width: double.infinity,
                decoration: BoxDecoration(
                  color: canaraYellow.withOpacity(0.18),
                  borderRadius: BorderRadius.circular(12),
                ),
                padding: const EdgeInsets.all(12),
                child: Row(
                  children: [
                    Icon(Icons.shield, color: canaraBlue, size: 32),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: const [
                          Text(
                            'You are eligible for an instant life cover of upto 5 lakhs',
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              color: Colors.black,
                              fontSize: 14,
                            ),
                          ),
                          SizedBox(height: 4),
                          Text(
                            'To know more visit: Canara HSBC Life Insurance section on your ai1 App',
                            style: TextStyle(
                              color: Colors.black87,
                              fontSize: 12,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _tabButton(String label, bool selected, Color color, int index) {
    return Expanded(
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: 4),
        decoration: BoxDecoration(
          color: selected ? color.withOpacity(0.12) : Colors.transparent,
          borderRadius: BorderRadius.circular(8),
        ),
        child: TextButton(
          onPressed: () {
            setState(() {
              _selectedTab = index;
            });
          },
          child: Text(
            label,
            style: TextStyle(
              color: selected ? color : Colors.grey,
              fontWeight: selected ? FontWeight.bold : FontWeight.normal,
              fontSize: 13,
            ),
          ),
        ),
      ),
    );
  }

  Widget _getTabContent(Color canaraBlue, Color canaraYellow, Color canaraLightBlue, Color canaraPurple) {
    if (_selectedTab == 0) {
      // My Beneficiary UI
      return Container(
        width: double.infinity,
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(18),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.04),
              blurRadius: 4,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.group, color: canaraBlue, size: 28),
                const SizedBox(width: 10),
                const Text(
                  'My Beneficiaries',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                ),
                const Spacer(),
                TextButton.icon(
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => AddBeneficiaryPage()),
                    );
                  },
                  icon: Icon(Icons.add, color: canaraBlue),
                  label: Text('Add', style: TextStyle(color: canaraBlue)),
                ),
              ],
            ),
            const Divider(height: 24),
            ListTile(
              leading: CircleAvatar(
                backgroundColor: canaraBlue.withOpacity(0.1),
                child: Icon(Icons.person, color: canaraBlue),
              ),
              title: const Text('Ravi Kumar'),
              subtitle: const Text('XXXXXX1234'),
              trailing: Icon(Icons.chevron_right, color: Colors.grey),
              onTap: () {},
            ),
            ListTile(
              leading: CircleAvatar(
                backgroundColor: canaraYellow.withOpacity(0.1),
                child: Icon(Icons.person, color: canaraYellow),
              ),
              title: const Text('Priya Sharma'),
              subtitle: const Text('XXXXXX5678'),
              trailing: Icon(Icons.chevron_right, color: Colors.grey),
              onTap: () {},
            ),
            ListTile(
              leading: CircleAvatar(
                backgroundColor: canaraLightBlue.withOpacity(0.1),
                child: Icon(Icons.person, color: canaraLightBlue),
              ),
              title: const Text('Amit Verma'),
              subtitle: const Text('XXXXXX9012'),
              trailing: Icon(Icons.chevron_right, color: Colors.grey),
              onTap: () {},
            ),
            const SizedBox(height: 8),
            Center(
              child: TextButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => MyBeneficiaryPage()),
                  );
                },
                child: Text('View All', style: TextStyle(color: canaraBlue)),
              ),
            ),
          ],
        ),
      );
    } else if (_selectedTab == 2) {
      // Frequently Used UI
      return Container(
        width: double.infinity,
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(18),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.04),
              blurRadius: 4,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(Icons.star, color: canaraYellow, size: 28),
                const SizedBox(width: 10),
                const Text(
                  'Frequently Used',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                ),
              ],
            ),
            const Divider(height: 24),
            ListTile(
              leading: CircleAvatar(
                backgroundColor: canaraBlue.withOpacity(0.1),
                child: Icon(Icons.send, color: canaraBlue),
              ),
              title: const Text('Send Money'),
              subtitle: const Text('To Ravi Kumar'),
              trailing: Icon(Icons.chevron_right, color: Colors.grey),
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => SendMoneyPage()),
                );
              },
            ),
            ListTile(
              leading: CircleAvatar(
                backgroundColor: canaraPurple.withOpacity(0.1),
                child: Icon(Icons.receipt_long, color: canaraPurple),
              ),
              title: const Text('Bill Pay'),
              subtitle: const Text('Electricity Bill'),
              trailing: Icon(Icons.chevron_right, color: Colors.grey),
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => const DirectPayPage()),
                );
              },
            ),
            ListTile(
              leading: CircleAvatar(
                backgroundColor: canaraLightBlue.withOpacity(0.1),
                child: Icon(Icons.account_balance_wallet, color: canaraLightBlue),
              ),
              title: const Text('Direct Pay'),
              subtitle: const Text('To Merchant'),
              trailing: Icon(Icons.chevron_right, color: Colors.grey),
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => const DirectPayPage()),
                );
              },
            ),
          ],
        ),
      );
    } else {
      // My Dashboard UI (default/original)
      return Container(
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(18),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.04),
              blurRadius: 4,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 8),
        child: Column(
          children: [
            GridView.count(
              crossAxisCount: 4,
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              mainAxisSpacing: 14,
              crossAxisSpacing: 8,
              childAspectRatio: 0.85,
              children: [
                _dashboardItem('Send Money', Icons.send, canaraBlue, () {
                  Navigator.push(context, MaterialPageRoute(builder: (context) => SendMoneyPage()));
                }),
                _dashboardItem('Direct Pay', Icons.payment, canaraYellow, () {
                  Navigator.push(context, MaterialPageRoute(builder: (context) => const DirectPayPage()));
                }),
                _dashboardItem('My Beneficiary', Icons.group, canaraLightBlue, () {
                  setState(() {
                    _selectedTab = 0;
                  });
                }),
                _dashboardItem('ePassbook', Icons.book, canaraPurple, () {
                  Navigator.push(context, MaterialPageRoute(builder: (context) => const EPassbookPage()));
                }),
                _dashboardItem('Bill Pay', Icons.receipt_long, canaraBlue, () {
                  Navigator.push(context, MaterialPageRoute(builder: (context) => const DirectPayPage()));
                }),
                _dashboardItem('Card-less Cash', Icons.atm, canaraYellow, () {
                  Navigator.push(context, MaterialPageRoute(builder: (context) => const CardlessCashPage()));
                }),
                _dashboardItem('Other Bank\nAccounts', Icons.account_balance, canaraLightBlue, () {
                  Navigator.push(context, MaterialPageRoute(builder: (context) => const HistoryPage()));
                }),
                _dashboardItem('History', Icons.history, canaraPurple, () {
                  Navigator.push(context, MaterialPageRoute(builder: (context) => const HistoryPage()));
                }),
                _dashboardItem('Manage\nAccounts', Icons.settings, canaraBlue, () {
                  Navigator.push(context, MaterialPageRoute(builder: (context) => const HistoryPage()));
                }),
              ],
            ),
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerRight,
              child: TextButton.icon(
                onPressed: () {},
                icon: Icon(Icons.expand_more, color: canaraBlue),
                label: Text('More', style: TextStyle(color: canaraBlue)),
              ),
            ),
          ],
        ),
      );
    }
  }

  Widget _dashboardItem(String label, IconData icon, Color color, VoidCallback onTap) {
    return GestureDetector(
      onTap: onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: color.withOpacity(0.12),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Center(
              child: Icon(
                icon,
                size: 26,
                color: color,
              ),
            ),
          ),
          const SizedBox(height: 6),
          Text(
            label,
            style: TextStyle(fontSize: 11, color: Colors.black, fontWeight: FontWeight.w500),
            textAlign: TextAlign.center,
            maxLines: 2,
          ),
        ],
      ),
    );
  }
}
