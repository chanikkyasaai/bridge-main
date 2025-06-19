import 'package:flutter/material.dart';

class ProfilePage extends StatelessWidget {
  const ProfilePage({super.key});

  @override
  Widget build(BuildContext context) {
    final Color canaraBlue = const Color(0xFF0072BC);
    final Color canaraYellow = const Color(0xFFFFD600);
    final Color canaraLightBlue = const Color(0xFF00B9F1);
    final Color canaraDarkBlue = const Color(0xFF003366);
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: const Text('Profile', style: TextStyle(color: Colors.black)),
        centerTitle: true,
        iconTheme: IconThemeData(color: canaraBlue),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 18.0, vertical: 12),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  CircleAvatar(
                    backgroundImage: AssetImage('assets/images/logo.jpeg'),
                    radius: 28,
                  ),
                  const SizedBox(width: 14),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: const [
                      Text('DEAR CUSTOMER', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18)),
                      SizedBox(height: 4),
                      Text('+9170XXXXXX78', style: TextStyle(color: Colors.grey)),
                    ],
                  ),
                ],
              ),
              const SizedBox(height: 18),
              _sectionTitle('Banking', canaraDarkBlue),
              _profileTile('De-Register', Icons.logout, canaraBlue),
              _profileTile('Manage View Balance/Statement', Icons.account_balance_wallet, canaraYellow),
              _profileTile('Change MPIN', Icons.lock, canaraLightBlue),
              _profileTile('Forgot MPIN', Icons.help_outline, canaraBlue),
              _profileTile('Change Passcode', Icons.password, canaraYellow),
              _profileTile('Get MMID', Icons.confirmation_number, canaraLightBlue),
              _profileTile('My Accounts', Icons.account_box, canaraBlue),
              _profileTile('MB Fund Transfer', Icons.swap_horiz, canaraYellow, trailing: _blockSwitch()),
              const SizedBox(height: 18),
              _sectionTitle('UPI', canaraDarkBlue),
              _profileTile('My UPI Accounts', Icons.account_balance, canaraBlue),
              _profileTile('MY UPI ID', Icons.alternate_email, canaraYellow),
            ],
          ),
        ),
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: 4,
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
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
      floatingActionButton: FloatingActionButton(
        backgroundColor: canaraBlue,
        onPressed: () {},
        child: const Icon(Icons.qr_code, color: Colors.white),
      ),
    );
  }

  Widget _sectionTitle(String title, Color color) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Text(title, style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: color)),
    );
  }

  Widget _profileTile(String label, IconData icon, Color color, {Widget? trailing}) {
    return ListTile(
      contentPadding: const EdgeInsets.symmetric(vertical: 0, horizontal: 0),
      leading: Icon(icon, color: color),
      title: Text(label, style: TextStyle(fontWeight: FontWeight.w500)),
      trailing: trailing,
      onTap: () {},
    );
  }

  Widget _blockSwitch() {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text('Block', style: TextStyle(color: Colors.grey, fontSize: 12)),
        Switch(value: false, onChanged: (v) {}),
        Text('Unblock', style: TextStyle(color: Colors.grey, fontSize: 12)),
      ],
    );
  }
}