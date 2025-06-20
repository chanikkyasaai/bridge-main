import 'package:flutter/material.dart';

class MyQrCodePage extends StatelessWidget {
  const MyQrCodePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('My QR Code', style: TextStyle(fontWeight: FontWeight.bold)),
        centerTitle: true,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.qr_code, size: 80, color: Colors.amber[800]),
              const SizedBox(height: 24),
              Container(
                color: Colors.grey[200],
                padding: const EdgeInsets.all(16),
                child: Image.asset(
                  'assets/images/sample_qr.png',
                  width: 180,
                  height: 180,
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) => Icon(Icons.qr_code_2, size: 180, color: Colors.grey[400]),
                ),
              ),
              const SizedBox(height: 18),
              const Text(
                'Scan this QR to pay me via UPI',
                style: TextStyle(fontSize: 16, color: Colors.grey),
              ),
              const SizedBox(height: 12),
              ElevatedButton.icon(
                icon: const Icon(Icons.share),
                label: const Text('Share QR'),
                onPressed: () {
                  // Add share logic
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}
