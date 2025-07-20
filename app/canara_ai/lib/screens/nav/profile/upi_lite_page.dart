import 'package:flutter/material.dart';

class UpiLitePage extends StatelessWidget {
  const UpiLitePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('UPI Lite', style: TextStyle(fontWeight: FontWeight.bold)),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.flash_on, size: 64, color: Colors.amber[700]),
            const SizedBox(height: 24),
            const Text(
              'UPI Lite',
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            const Text(
              'Enjoy faster and smaller value UPI payments with UPI Lite. No PIN required for transactions up to ₹200!',
              style: TextStyle(fontSize: 16, color: Colors.grey),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              icon: const Icon(Icons.account_balance_wallet),
              label: const Text('Activate UPI Lite'),
              onPressed: () {
                // Add activation logic
              },
            ),
          ],
        ),
      ),
    );
  }
}
