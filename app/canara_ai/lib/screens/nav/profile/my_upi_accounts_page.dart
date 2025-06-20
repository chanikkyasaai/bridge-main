import 'package:flutter/material.dart';

class MyUpiAccountsPage extends StatelessWidget {
  const MyUpiAccountsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('My UPI Accounts', style: TextStyle(fontWeight: FontWeight.bold)),
        centerTitle: true,
      ),
      body: ListView(
        padding: const EdgeInsets.all(24.0),
        children: [
          Icon(Icons.account_balance, size: 64, color: Colors.blue[800]),
          const SizedBox(height: 24),
          const Text(
            'Linked UPI Accounts',
            style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 12),
          Card(
            elevation: 2,
            child: ListTile(
              leading: Icon(Icons.account_balance_wallet, color: Colors.teal[700]),
              title: const Text('Savings Account'),
              subtitle: const Text('XXXXXX1234'),
              trailing: const Icon(Icons.check_circle, color: Colors.green),
            ),
          ),
          Card(
            elevation: 2,
            child: ListTile(
              leading: Icon(Icons.account_balance_wallet, color: Colors.orange[700]),
              title: const Text('Current Account'),
              subtitle: const Text('XXXXXX5678'),
              trailing: const Icon(Icons.check_circle, color: Colors.green),
            ),
          ),
        ],
      ),
    );
  }
}
