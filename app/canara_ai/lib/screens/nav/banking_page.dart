import 'package:flutter/material.dart';

class BankingPage extends StatefulWidget {
  const BankingPage({super.key});

  @override
  State<BankingPage> createState() => _BankingPageState();
}

class _BankingPageState extends State<BankingPage> {
  int _selectedTab = 1; // 0: My Beneficiary, 1: My Dashboard, 2: Frequently Used

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
                        Image.asset(
                          'assets/images/app_icon.png', // Replace with your ai logo asset
                          height: 34,
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
                    const SizedBox(height: 10),
                    Row(
                      children: [
                        Icon(Icons.remove_red_eye, color: Colors.white, size: 22),
                        const SizedBox(width: 8),
                        Flexible(
                          child: Text(
                            'View Balance',
                            style: TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.bold,
                              fontSize: 15,
                            ),
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                        const Spacer(),
                        TextButton(
                          onPressed: () {},
                          child: const Text('Statement', style: TextStyle(color: Colors.white)),
                        ),
                        TextButton(
                          onPressed: () {},
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
                  onPressed: () {},
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
                onPressed: () {},
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
              onTap: () {},
            ),
            ListTile(
              leading: CircleAvatar(
                backgroundColor: canaraPurple.withOpacity(0.1),
                child: Icon(Icons.receipt_long, color: canaraPurple),
              ),
              title: const Text('Bill Pay'),
              subtitle: const Text('Electricity Bill'),
              trailing: Icon(Icons.chevron_right, color: Colors.grey),
              onTap: () {},
            ),
            ListTile(
              leading: CircleAvatar(
                backgroundColor: canaraLightBlue.withOpacity(0.1),
                child: Icon(Icons.account_balance_wallet, color: canaraLightBlue),
              ),
              title: const Text('Direct Pay'),
              subtitle: const Text('To Merchant'),
              trailing: Icon(Icons.chevron_right, color: Colors.grey),
              onTap: () {},
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
                _dashboardItem('assets/icons/bank.png', 'Send Money', canaraBlue),
                _dashboardItem('assets/icons/money.png', 'Direct Pay', canaraYellow),
                _dashboardItem('assets/icons/mobilephone.png', 'My Beneficiary', canaraLightBlue),
                _dashboardItem('assets/icons/passbook.png', 'ePassbook', canaraPurple),
                _dashboardItem('assets/icons/b.png', 'Bill Pay', canaraBlue),
                _dashboardItem('assets/icons/send-money.png', 'Card-less Cash', canaraYellow),
                _dashboardItem('assets/icons/bank.png', 'Other Bank Accounts', canaraLightBlue),
                _dashboardItem('assets/icons/folder.png', 'History', canaraPurple),
                _dashboardItem('assets/icons/mobile-banking.png', 'Manage Accounts', canaraBlue),
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

  Widget _dashboardItem(String asset, String label, Color color) {
    return Column(
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
            child: Image.asset(
              asset,
              height: 26,
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
    );
  }
}
