import 'package:flutter/material.dart';

class ManageAccountPage extends StatelessWidget {
  const ManageAccountPage({super.key});

  @override
  Widget build(BuildContext context) {
    final Color canaraBlue = const Color(0xFF0072BC);
    final Color canaraLightBlue = const Color(0xFF00B9F1);
    final Color canaraYellow = const Color(0xFFFFD600);
    
    return Scaffold(
      backgroundColor: const Color(0xFFF7F9FB),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: const Text('Manage Account', style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold)),
        iconTheme: IconThemeData(color: canaraBlue),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                gradient: LinearGradient(colors: [canaraBlue, canaraLightBlue]),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Savings Account', style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
                  const SizedBox(height: 8),
                  const Text('Account: 110084150765', style: TextStyle(color: Colors.white)),
                  const SizedBox(height: 4),
                  const Text('IFSC: CNRB0001234', style: TextStyle(color: Colors.white)),
                ],
              ),
            ),
            const SizedBox(height: 20),
            Expanded(
              child: ListView(
                children: [
                  _manageOption(Icons.credit_card, 'Debit Card Management', 'Block/Unblock, Set PIN', canaraBlue, () {}),
                  _manageOption(Icons.phone, 'Mobile Banking', 'Update mobile number', canaraLightBlue, () {}),
                  _manageOption(Icons.email, 'Email Updates', 'Manage email notifications', canaraYellow, () {}),
                  _manageOption(Icons.security, 'Security Settings', 'Change MPIN, Login PIN', canaraBlue, () {}),
                  _manageOption(Icons.account_balance, 'Account Limits', 'Transaction limits', canaraLightBlue, () {}),
                  _manageOption(Icons.notifications, 'Alerts & Notifications', 'SMS, Email alerts', canaraYellow, () {}),
                  _manageOption(Icons.download, 'Cheque Book Request', 'Order new cheque book', canaraBlue, () {}),
                  _manageOption(Icons.description, 'Account Certificate', 'Download certificates', canaraLightBlue, () {}),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _manageOption(IconData icon, String title, String subtitle, Color color, VoidCallback onTap) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 4, offset: const Offset(0, 2))],
      ),
      child: ListTile(
        contentPadding: const EdgeInsets.all(16),
        leading: Container(
          width: 48,
          height: 48,
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(icon, color: color, size: 24),
        ),
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.w500)),
        subtitle: Text(subtitle),
        trailing: Icon(Icons.chevron_right, color: Colors.grey[400]),
        onTap: onTap,
      ),
    );
  }
}
