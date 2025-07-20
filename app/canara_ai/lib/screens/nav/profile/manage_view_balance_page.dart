import 'package:flutter/material.dart';

class ManageViewBalancePage extends StatelessWidget {
  const ManageViewBalancePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Manage View Balance', style: TextStyle(fontWeight: FontWeight.bold)),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(Icons.account_balance_wallet, size: 64, color: Colors.amber[800]),
            const SizedBox(height: 24),
            const Text(
              'View & Manage Balances',
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            Card(
              elevation: 2,
              child: ListTile(
                leading: Icon(Icons.account_balance, color: Colors.blue[700]),
                title: const Text('Savings Account'),
                subtitle: const Text('XXXXXX1234'),
                trailing: const Text('₹ 25,000', style: TextStyle(fontWeight: FontWeight.bold)),
              ),
            ),
            Card(
              elevation: 2,
              child: ListTile(
                leading: Icon(Icons.account_balance, color: Colors.green[700]),
                title: const Text('Current Account'),
                subtitle: const Text('XXXXXX5678'),
                trailing: const Text('₹ 1,20,000', style: TextStyle(fontWeight: FontWeight.bold)),
              ),
            ),
            const SizedBox(height: 18),
            ElevatedButton.icon(
              icon: const Icon(Icons.visibility),
              label: const Text('View Statement'),
              onPressed: () {
                // Add view statement logic
              },
            ),
          ],
        ),
      ),
    );
  }
}
