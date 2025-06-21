import 'package:flutter/material.dart';

import 'banking_page.dart';
import 'profile/deregister_page.dart';
import 'profile/fund_transfer_page.dart';
import 'profile/manage_view_balance_page.dart';
import 'profile/my_accounts_page.dart';
import 'profile/my_qr_code_page.dart';
import 'profile/my_upi_accounts_page.dart';
import 'profile/my_upi_number_page.dart';
import 'profile/upi_lite_page.dart';

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
                _profileTile('Banking', Icons.account_balance, canaraBlue, onTap: () {
                  Navigator.push(context, MaterialPageRoute(builder: (_) => const BankingPage()));
                }),
                _profileTile('De-Register', Icons.logout, canaraBlue, onTap: () {
                  Navigator.push(context, MaterialPageRoute(builder: (_) => const DeRegisterPage()));
                }),
                _profileTile('Manage View Balance/Statement', Icons.account_balance_wallet, canaraYellow, onTap: () {
                  Navigator.push(context, MaterialPageRoute(builder: (_) => const ManageViewBalancePage()));
                }),
                _profileTile('Change MPIN', Icons.lock, canaraLightBlue, onTap: () => _showChangeMpinDialog(context)),
                _profileTile('Forgot MPIN', Icons.help_outline, canaraBlue, onTap: () => _showForgotMpinDialog(context)),
                _profileTile('Change Passcode', Icons.password, canaraYellow, onTap: () => _showChangePasscodeDialog(context)),
                _profileTile('Get MMID', Icons.confirmation_number, canaraLightBlue, onTap: () => _showGetMmidDialog(context)),
                _profileTile('My Accounts', Icons.account_box, canaraBlue, onTap: () {
                  Navigator.push(context, MaterialPageRoute(builder: (_) => const MyAccountsPage()));
                }),
                _profileTile('MB Fund Transfer', Icons.swap_horiz, canaraYellow, trailing: _blockSwitch(), onTap: () {
                  Navigator.push(context, MaterialPageRoute(builder: (_) => const FundTransferPage()));
                }),
                const SizedBox(height: 18),
                _sectionTitle('UPI', canaraDarkBlue),
                _profileTile('My UPI Accounts', Icons.person_outline, canaraBlue, onTap: () {
                  Navigator.push(context, MaterialPageRoute(builder: (_) => const MyUpiAccountsPage()));
                }),
                _profileTile('My QR Code', Icons.qr_code, canaraYellow, onTap: () {
                  Navigator.push(context, MaterialPageRoute(builder: (_) => const MyQrCodePage()));
                }),
                _profileTile('Blocked Users', Icons.block, canaraLightBlue, onTap: () => _showBlockedUsersDialog(context)),
                _profileTile('Block/Unblock UPI Services', Icons.lock_open, canaraBlue, onTap: () => _showBlockUnblockUpiDialog(context)),
                _profileTile('My UPI Number', Icons.numbers, canaraYellow, onTap: () {
                  Navigator.push(context, MaterialPageRoute(builder: (_) => const MyUpiNumberPage()));
                }),
                _profileTile('UPI Lite', Icons.flash_on, canaraLightBlue, onTap: () {
                  Navigator.push(context, MaterialPageRoute(builder: (_) => const UpiLitePage()));
                }),
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

  Widget _profileTile(String label, IconData icon, Color color, {Widget? trailing, void Function()? onTap}) {
    return ListTile(
      contentPadding: const EdgeInsets.symmetric(vertical: 0, horizontal: 0),
      leading: Icon(icon, color: color),
      title: Text(
        label,
        style: const TextStyle(fontWeight: FontWeight.w500),
        overflow: TextOverflow.ellipsis,
      ),
      trailing: trailing,
      onTap: onTap ?? () {},
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

  void _showChangeMpinDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
        title: Row(
          children: [
            Icon(Icons.lock, color: Color(0xFF0072BC)),
            const SizedBox(width: 8),
            const Text('Change MPIN', style: TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'Enter your current and new MPIN to proceed.',
              style: TextStyle(fontSize: 15, color: Colors.black87),
            ),
            const SizedBox(height: 18),
            TextField(
              obscureText: true,
              decoration: InputDecoration(
                labelText: 'Current MPIN',
                prefixIcon: const Icon(Icons.vpn_key),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                filled: true,
                fillColor: Colors.blue[50],
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              obscureText: true,
              decoration: InputDecoration(
                labelText: 'New MPIN',
                prefixIcon: const Icon(Icons.lock_outline),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                filled: true,
                fillColor: Colors.blue[50],
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            child: const Text('Cancel', style: TextStyle(color: Colors.grey)),
            onPressed: () => Navigator.pop(context),
          ),
          ElevatedButton.icon(
            icon: const Icon(Icons.check_circle_outline),
            label: const Text('Change'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF0072BC),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
            ),
            onPressed: () {/* Add logic */},
          ),
        ],
      ),
    );
  }

  void _showForgotMpinDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
        title: Row(
          children: [
            Icon(Icons.help_outline, color: Color(0xFF0072BC)),
            const SizedBox(width: 8),
            const Text('Forgot MPIN', style: TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
        content: const Text(
          'To reset your MPIN, you will receive an OTP on your registered mobile number.\n\nDo you want to proceed?',
          style: TextStyle(fontSize: 15, color: Colors.black87),
        ),
        actions: [
          TextButton(
            child: const Text('Cancel', style: TextStyle(color: Colors.grey)),
            onPressed: () => Navigator.pop(context),
          ),
          ElevatedButton.icon(
            icon: const Icon(Icons.sms),
            label: const Text('Proceed'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF0072BC),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
            ),
            onPressed: () {/* Add logic */},
          ),
        ],
      ),
    );
  }

  void _showChangePasscodeDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
        title: Row(
          children: [
            Icon(Icons.password, color: Color(0xFFFFD600)),
            const SizedBox(width: 8),
            const Text('Change Passcode', style: TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'Enter your current and new passcode.',
              style: TextStyle(fontSize: 15, color: Colors.black87),
            ),
            const SizedBox(height: 18),
            TextField(
              obscureText: true,
              decoration: InputDecoration(
                labelText: 'Current Passcode',
                prefixIcon: const Icon(Icons.vpn_key),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                filled: true,
                fillColor: Colors.yellow[50],
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              obscureText: true,
              decoration: InputDecoration(
                labelText: 'New Passcode',
                prefixIcon: const Icon(Icons.lock_outline),
                border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                filled: true,
                fillColor: Colors.yellow[50],
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            child: const Text('Cancel', style: TextStyle(color: Colors.grey)),
            onPressed: () => Navigator.pop(context),
          ),
          ElevatedButton.icon(
            icon: const Icon(Icons.check_circle_outline),
            label: const Text('Change'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFFFD600),
              foregroundColor: Colors.black,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
            ),
            onPressed: () {/* Add logic */},
          ),
        ],
      ),
    );
  }

  void _showGetMmidDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
        title: Row(
          children: [
            Icon(Icons.confirmation_number, color: Color(0xFF00B9F1)),
            const SizedBox(width: 8),
            const Text('Get MMID', style: TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'Your MMID for mobile banking is:',
              style: TextStyle(fontSize: 15, color: Colors.black87),
            ),
            const SizedBox(height: 18),
            Container(
              padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 24),
              decoration: BoxDecoration(
                color: Colors.blue[50],
                borderRadius: BorderRadius.circular(12),
              ),
              child: const SelectableText(
                '1234 5678',
                style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, letterSpacing: 2),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            child: const Text('Close', style: TextStyle(color: Colors.grey)),
            onPressed: () => Navigator.pop(context),
          ),
          ElevatedButton.icon(
            icon: const Icon(Icons.copy),
            label: const Text('Copy'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF00B9F1),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
            ),
            onPressed: () {/* Add copy logic */},
          ),
        ],
      ),
    );
  }

  void _showBlockedUsersDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
        title: Row(
          children: [
            Icon(Icons.block, color: Color(0xFF00B9F1)),
            const SizedBox(width: 8),
            const Text('Blocked Users', style: TextStyle(fontWeight: FontWeight.bold)),
          ],
        ),
        content: SizedBox(
          width: double.maxFinite,
          child: ListView(
            shrinkWrap: true,
            children: [
              Card(
                color: Colors.red[50],
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                child: const ListTile(
                  leading: Icon(Icons.person_off, color: Colors.red),
                  title: Text('user1@upi'),
                  trailing: Icon(Icons.delete, color: Colors.grey),
                ),
              ),
              Card(
                color: Colors.red[50],
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                child: const ListTile(
                  leading: Icon(Icons.person_off, color: Colors.red),
                  title: Text('user2@upi'),
                  trailing: Icon(Icons.delete, color: Colors.grey),
                ),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            child: const Text('Close', style: TextStyle(color: Colors.grey)),
            onPressed: () => Navigator.pop(context),
          ),
        ],
      ),
    );
  }

  void _showBlockUnblockUpiDialog(BuildContext context) {
    bool isBlocked = false;
    showDialog(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setState) => AlertDialog(
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
          title: Row(
            children: [
              Icon(Icons.lock_open, color: Color(0xFF0072BC)),
              const SizedBox(width: 8),
              const Text('Block/Unblock UPI Services', style: TextStyle(fontWeight: FontWeight.bold)),
            ],
          ),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(
                children: [
                  const Text('Status: ', style: TextStyle(fontWeight: FontWeight.w500)),
                  Chip(
                    label: Text(isBlocked ? 'Blocked' : 'Active'),
                    backgroundColor: isBlocked ? Colors.red[100] : Colors.green[100],
                    labelStyle: TextStyle(color: isBlocked ? Colors.red : Colors.green),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Text(
                isBlocked ? 'Your UPI services are currently blocked. Tap Unblock to reactivate.' : 'Your UPI services are active. Tap Block to restrict UPI transactions.',
                style: const TextStyle(fontSize: 14, color: Colors.black87),
              ),
            ],
          ),
          actions: [
            TextButton(
              child: const Text('Cancel', style: TextStyle(color: Colors.grey)),
              onPressed: () => Navigator.pop(context),
            ),
            ElevatedButton.icon(
              icon: Icon(isBlocked ? Icons.lock_open : Icons.lock),
              label: Text(isBlocked ? 'Unblock' : 'Block'),
              style: ElevatedButton.styleFrom(
                backgroundColor: isBlocked ? Colors.green : Colors.red,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
              ),
              onPressed: () {
                setState(() => isBlocked = !isBlocked);
                // Add block/unblock logic
              },
            ),
          ],
        ),
      ),
    );
  }
}
