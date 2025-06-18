import 'package:flutter/material.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _selectedIndex = 0;

  final Color canaraBlue = const Color(0xFF0072BC);
  final Color canaraYellow = const Color(0xFFFFD600);
  final Color canaraLightBlue = const Color(0xFF00B9F1);
  final Color canaraDarkBlue = const Color(0xFF003366);
  final Color canaraPurple = const Color(0xFF7B1FA2);

  void _onNavTap(int index) {
    setState(() {
      _selectedIndex = index;
    });
    // Navigation logic can be added here
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: SingleChildScrollView(
          child: Column(
            children: [
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(vertical: 18),
                color: Colors.transparent,
                child: Text(
                  'One stop solution for all your Banking needs',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: canaraDarkBlue),
                ),
              ),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16.0),
                child: Column(
                  children: [
                    Row(
                      children: [
                        CircleAvatar(
                          backgroundImage: AssetImage('assets/images/logo.jpeg'),
                          radius: 18,
                        ),
                        const SizedBox(width: 10),
                        const Text('DEAR CUSTOMER', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                        const Spacer(),
                        Icon(Icons.search, color: canaraBlue),
                        const SizedBox(width: 12),
                        Icon(Icons.notifications_none, color: canaraBlue),
                      ],
                    ),
                    const SizedBox(height: 18),
                    // Portfolio Card
                    Container(
                      width: double.infinity,
                      decoration: BoxDecoration(
                        gradient: LinearGradient(colors: [canaraBlue, canaraLightBlue]),
                        borderRadius: BorderRadius.circular(18),
                      ),
                      padding: const EdgeInsets.all(18),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Image.asset('assets/images/app.webp', height: 32),
                              const SizedBox(width: 8),
                              Text('ai', style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: canaraYellow)),
                              const Spacer(),
                              Image.asset('assets/images/logo.jpeg', height: 28),
                            ],
                          ),
                          const SizedBox(height: 10),
                          Text('My Portfolio', style: TextStyle(fontSize: 18, color: Colors.white, fontWeight: FontWeight.bold)),
                          const SizedBox(height: 12),
                          Wrap(
                            spacing: 18,
                            runSpacing: 12,
                            children: [
                              _portfolioTile(Icons.account_balance_wallet, 'Savings (A/C)', canaraPurple),
                              _portfolioTile(Icons.account_balance, 'OD Account (A/C)', canaraBlue),
                              _portfolioTile(Icons.savings, 'Deposits (A/C)', canaraYellow),
                              _portfolioTile(Icons.request_page, 'Loans (A/C)', canaraLightBlue),
                            ],
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 18),
                    // Banner
                    Container(
                      width: double.infinity,
                      decoration: BoxDecoration(
                        color: canaraYellow.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      padding: const EdgeInsets.all(12),
                      child: Row(
                        children: [
                          Icon(Icons.pedal_bike, color: canaraBlue),
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
                    const SizedBox(height: 18),
                    // Pay & Transfer
                    Align(
                      alignment: Alignment.centerLeft,
                      child: Text('Pay & Transfer', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: canaraDarkBlue)),
                    ),
                    const SizedBox(height: 10),
                    Wrap(
                      spacing: 18,
                      runSpacing: 12,
                      children: [
                        _payTile(Icons.send, 'Send Money', canaraBlue),
                        _payTile(Icons.payment, 'Direct Pay', canaraYellow),
                        _payTile(Icons.people, 'My Beneficiary', canaraLightBlue),
                        _payTile(Icons.book, 'ePassbook', canaraPurple),
                        _payTile(Icons.money_off, 'Card-less Cash', canaraBlue),
                        _payTile(Icons.volunteer_activism, 'Donation', canaraYellow),
                        _payTile(Icons.history, 'History', canaraLightBlue),
                        _payTile(Icons.manage_accounts, 'Manage Accounts', canaraPurple),
                      ],
                    ),
                    const SizedBox(height: 18),
                    // UPI ID
                    Container(
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
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
      floatingActionButton: FloatingActionButton(
        backgroundColor: canaraBlue,
        onPressed: () {},
        child: const Icon(Icons.qr_code, color: Colors.white),
      ),
      bottomNavigationBar: BottomAppBar(
        shape: const CircularNotchedRectangle(),
        notchMargin: 8,
        child: BottomNavigationBar(
          currentIndex: _selectedIndex,
          onTap: _onNavTap,
          selectedItemColor: canaraBlue,
          unselectedItemColor: Colors.grey,
          showUnselectedLabels: true,
          items: const [
            BottomNavigationBarItem(icon: Icon(Icons.apps), label: 'All'),
            BottomNavigationBarItem(icon: Icon(Icons.account_balance), label: 'Bank'),
            BottomNavigationBarItem(icon: Icon(Icons.qr_code), label: ''),
            BottomNavigationBarItem(icon: Icon(Icons.history), label: 'History'),
            BottomNavigationBarItem(icon: Icon(Icons.person), label: 'Profile'),
          ],
        ),
      ),
    );
  }

  Widget _portfolioTile(IconData icon, String label, Color color) {
    return Container(
      width: 120,
      height: 60,
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, color: color, size: 28),
          const SizedBox(width: 8),
          Flexible(
            child: Text(label, style: TextStyle(fontSize: 13, color: Colors.white, fontWeight: FontWeight.bold)),
          ),
        ],
      ),
    );
  }

  Widget _payTile(IconData icon, String label, Color color) {
    return Container(
      width: 120,
      height: 60,
      decoration: BoxDecoration(
        color: color.withOpacity(0.10),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, color: color, size: 28),
          const SizedBox(height: 4),
          Text(label, style: TextStyle(fontSize: 13, color: color, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}