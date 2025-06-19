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
        title: const Text('Profile', style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold)),
        centerTitle: true,
        iconTheme: IconThemeData(color: canaraBlue),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
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
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: const [
                          Text(
                            'DEAR CUSTOMER',
                            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
                            overflow: TextOverflow.ellipsis,
                          ),
                          SizedBox(height: 4),
                          Text(
                            '+9170XXXXXX78',
                            style: TextStyle(color: Colors.grey),
                            overflow: TextOverflow.ellipsis,
                          ),
                        ],
                      ),
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
                _profileTile('My UPI Accounts', Icons.person_outline, canaraBlue),
                _profileTile('My QR Code', Icons.qr_code, canaraYellow),
                _profileTile('Blocked Users', Icons.block, canaraLightBlue),
                _profileTile('Block/Unblock UPI Services', Icons.lock_open, canaraBlue),
                _profileTile('My UPI Number', Icons.numbers, canaraYellow),
                _profileTile('UPI Lite', Icons.flash_on, canaraLightBlue),
                const SizedBox(height: 18),
                _sectionTitle('General', canaraDarkBlue),
                _profileTile('Change Language', Icons.language, canaraBlue),
                _profileTile('App Permissions', Icons.privacy_tip, canaraYellow),
                _profileTile('Refer A Friend', Icons.share, canaraLightBlue),
                _profileTile('Manage Profile', Icons.manage_accounts, canaraBlue),
                _profileTile('Validate App Version', Icons.verified, canaraYellow),
                ListTile(
                  leading: Icon(Icons.brightness_6, color: canaraLightBlue),
                  title: const Text('Theme', style: TextStyle(fontWeight: FontWeight.w500)),
                  trailing: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Text('Light'),
                      Switch(value: false, onChanged: (v) {}),
                      const Text('Dark'),
                    ],
                  ),
                ),
                ListTile(
                  leading: Icon(Icons.apps, color: canaraBlue),
                  title: const Text('All Services', style: TextStyle(fontWeight: FontWeight.w500)),
                  trailing: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Text('One Row'),
                      Switch(value: true, onChanged: (v) {}),
                      const Text('Two Rows'),
                    ],
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 8.0),
                  child: SingleChildScrollView(
                    scrollDirection: Axis.horizontal,
                    child: Row(
                      children: [
                        Icon(Icons.palette, color: canaraYellow),
                        const SizedBox(width: 16),
                        const Text('Icons'),
                        const SizedBox(width: 16),
                        Radio(value: true, groupValue: true, onChanged: (v) {}),
                        const Text('Blue'),
                        Radio(value: false, groupValue: true, onChanged: (v) {}),
                        const Text('Purple'),
                        Radio(value: false, groupValue: true, onChanged: (v) {}),
                        const Text('Green'),
                        Radio(value: false, groupValue: true, onChanged: (v) {}),
                        const Text('Pink'),
                      ],
                    ),
                  ),
                ),
                _profileTile('Change Transaction SMS Language', Icons.sms, canaraBlue),
                const SizedBox(height: 18),
                _sectionTitle('Help', canaraDarkBlue),
                _profileTile('Relationship Manager Details', Icons.person_search, canaraBlue),
                _profileTile('Feedback', Icons.feedback, canaraYellow),
                _profileTile('Rate Us', Icons.star_rate, canaraLightBlue),
                _profileTile('Contact Us', Icons.phone, canaraBlue),
                _profileTile('Cybercrime Reporting Portal', Icons.security, canaraYellow),
                _profileTile('National Cybercrime Helpline', Icons.support_agent, canaraLightBlue),
                _profileTile('Locate Us', Icons.location_on, canaraBlue),
                const SizedBox(height: 18),
                _sectionTitle('Follow Us', canaraDarkBlue),
                SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Row(
                    children: [
                      IconButton(icon: Icon(Icons.facebook, color: Colors.blue), onPressed: () {}),
                      IconButton(icon: Icon(Icons.alternate_email, color: Colors.red), onPressed: () {}), // Instagram substitute
                      IconButton(icon: Icon(Icons.video_library, color: Colors.red), onPressed: () {}), // YouTube substitute
                      IconButton(icon: Icon(Icons.business, color: Colors.blue[800]), onPressed: () {}), // LinkedIn substitute
                      IconButton(icon: Icon(Icons.push_pin, color: Colors.redAccent), onPressed: () {}), // Pinterest substitute
                      IconButton(icon: Icon(Icons.language, color: Colors.blueGrey), onPressed: () {}), // Website
                    ],
                  ),
                ),
                const SizedBox(height: 18),
                Center(
                  child: Text(
                    'App Version: 3.6.0',
                    style: TextStyle(color: Colors.grey),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
                const SizedBox(height: 8),
                Center(
                  child: TextButton.icon(
                    onPressed: () {},
                    icon: Icon(Icons.logout, color: canaraBlue),
                    label: const Text('Log Out', style: TextStyle(color: Colors.black)),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _sectionTitle(String title, Color color) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Text(
        title,
        style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: color),
        overflow: TextOverflow.ellipsis,
      ),
    );
  }

  Widget _profileTile(String label, IconData icon, Color color, {Widget? trailing}) {
    return ListTile(
      contentPadding: const EdgeInsets.symmetric(vertical: 0, horizontal: 0),
      leading: Icon(icon, color: color),
      title: Text(
        label,
        style: const TextStyle(fontWeight: FontWeight.w500),
        overflow: TextOverflow.ellipsis,
      ),
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
