import 'package:flutter/material.dart';

class FundTransferPage extends StatelessWidget {
  const FundTransferPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Fund Transfer', style: TextStyle(fontWeight: FontWeight.bold)),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(Icons.swap_horiz, size: 64, color: Colors.amber[700]),
            const SizedBox(height: 24),
            const Text(
              'Transfer Funds',
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 18),
            TextField(
              decoration: const InputDecoration(
                labelText: 'Recipient Account Number',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              decoration: const InputDecoration(
                labelText: 'Amount',
                border: OutlineInputBorder(),
                prefixText: '₹ ',
              ),
              keyboardType: TextInputType.number,
            ),
            const SizedBox(height: 12),
            TextField(
              decoration: const InputDecoration(
                labelText: 'Remarks (optional)',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 24),
            Center(
              child: ElevatedButton.icon(
                icon: const Icon(Icons.send),
                label: const Text('Transfer'),
                onPressed: () {
                  // Add fund transfer logic
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
