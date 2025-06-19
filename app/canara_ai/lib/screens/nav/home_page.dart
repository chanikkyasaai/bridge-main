import 'package:canara_ai/screens/nav/banking_page.dart';
import 'package:canara_ai/screens/nav/cards_page.dart';
import 'package:canara_ai/screens/nav/profile_page.dart';
import 'package:canara_ai/widgets/notification_sheet.dart';
import 'package:canara_ai/widgets/qr_scan_sheet.dart';
import 'package:canara_ai/widgets/search_sheet.dart';
import 'package:flutter/material.dart';

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
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(18)),
      ),
      builder: (context) {
        return SearchSheet(
          canaraBlue: canaraBlue,
          canaraDarkBlue: canaraDarkBlue,
        );
      },
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
              ),
              padding: const EdgeInsets.all(18),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // ai + logo row
                  Row(
                    children: [
                      Image.asset('assets/images/app.webp', height: 32),
                      const SizedBox(width: 8),
                      const Spacer(),
                      Image.asset('assets/images/logo.jpeg', height: 28),
                    ],
                  ),
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      Text(
                        'My Portfolio',
                        style: TextStyle(fontSize: 18, color: Colors.white, fontWeight: FontWeight.bold),
                      ),
                      const Spacer(),
                      Switch(
                        value: _showBalance,
                        onChanged: (val) => setState(() => _showBalance = val),
                        activeColor: canaraYellow,
                      ),
                      Text(
                        _showBalance ? 'Show' : 'Hide',
                        style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      _portfolioTile('assets/icons/rupee.png', 'Savings (A/C)'),
                      _portfolioTile('assets/icons/save.png', 'OD Account (A/C)'),
                      _portfolioTile('assets/icons/donation.png', 'Deposits (A/C)'),
                      _portfolioTile('assets/icons/loan.png', 'Loans (A/C)'),
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
            _serviceItem('assets/icons/bank.png', 'Send Money'),
            _serviceItem('assets/icons/money.png', 'Direct Pay'),
            _serviceItem('assets/icons/mobilephone.png', 'My Beneficiary'),
            _serviceItem('assets/icons/passbook.png', 'ePassbook'),
            _serviceItem('assets/icons/send-money.png', 'Card-less Cash'),
            _serviceItem('assets/icons/contact-book.png', 'Donation'),
            _serviceItem('assets/icons/folder.png', 'History'),
            _serviceItem('assets/icons/mobile-banking.png', 'Manage Accounts'),
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
          // GST & Other Taxes
          _sectionTitle('GST & Other Taxes'),
          _serviceGrid([
            _serviceItem('assets/icons/money-1.png', 'Pay GST'),
            _serviceItem('assets/icons/money-1.png', 'Check GST Payment Status'),
            _serviceItem('assets/icons/money-1.png', 'Generate Receipt'),
            _serviceItem('assets/icons/computer.png', 'Online Tax Payment'),
          ]),
          // Invest & Insure
          _sectionTitle('Invest & Insure'),
          _serviceGrid([
            _serviceItem('assets/icons/life-insurance.png', 'Canara HSBC Life Insurance'),
            _serviceItem('assets/icons/rupee-1.png', 'ASBA'),
            _serviceItem('assets/icons/chart-file.png', 'Demat/Trade'),
            _serviceItem('assets/icons/google-docs.png', '26AS'),
            _serviceItem('assets/icons/rupees.png', 'Mutual Fund'),
            _serviceItem('assets/icons/insurance.png', 'Home Insurance'),
            _serviceItem('assets/icons/healthcare.png', 'Health Insurance'),
            _serviceItem('assets/icons/protection.png', 'Motor Insurance'),
          ]),
          const SizedBox(height: 24),
        ],
      ),
    );
  }

  Widget _portfolioTile(String asset, String label) {
    return Column(
      children: [
        Container(
          width: 48,
          height: 48,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.18),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Center(child: Image.asset(asset, height: 28)),
        ),
        const SizedBox(height: 6),
        SizedBox(
          width: 70,
          child: Text(
            label,
            style: const TextStyle(fontSize: 12, color: Colors.white, fontWeight: FontWeight.bold),
            textAlign: TextAlign.center,
            maxLines: 2,
          ),
        ),
      ],
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

  Widget _serviceItem(String asset, String label) {
    return Column(
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
              child: Image.asset(
            asset,
            height: 26,
            color: canaraBlue,
          )),
        ),
        const SizedBox(height: 6),
        Text(
          label,
          style: TextStyle(fontSize: 11, color: canaraDarkBlue, fontWeight: FontWeight.w500),
          textAlign: TextAlign.center,
          maxLines: 2,
        ),
      ],
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