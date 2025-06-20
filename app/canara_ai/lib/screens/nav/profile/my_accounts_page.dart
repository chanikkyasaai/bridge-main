import 'package:flutter/material.dart';

class MyAccountsPage extends StatelessWidget {
  const MyAccountsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('My Accounts', style: TextStyle(fontWeight: FontWeight.bold)),
        centerTitle: true,
      ),
      body: ListView(
        padding: const EdgeInsets.all(24.0),
        children: [
          Icon(Icons.account_box, size: 64, color: Colors.blue[900]),
          const SizedBox(height: 24),
          const Text(
            'Your Accounts',
            style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 12),
          Card(
            elevation: 2,
            child: ListTile(
              leading: Icon(Icons.savings, color: Colors.teal[700]),
              title: const Text('Savings Account'),
              subtitle: const Text('Account No: XXXXXX1234'),
              trailing: const Icon(Icons.arrow_forward_ios),
              onTap: () {},
            ),
          ),
          Card(
            elevation: 2,
            child: ListTile(
              leading: Icon(Icons.business_center, color: Colors.orange[700]),
              title: const Text('Current Account'),
              subtitle: const Text('Account No: XXXXXX5678'),
              trailing: const Icon(Icons.arrow_forward_ios),
              onTap: () {},
            ),
          ),
        ],
      ),
    );
  }
}
