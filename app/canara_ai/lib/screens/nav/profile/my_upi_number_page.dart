import 'package:flutter/material.dart';

class MyUpiNumberPage extends StatelessWidget {
  const MyUpiNumberPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('My UPI Number', style: TextStyle(fontWeight: FontWeight.bold)),
        centerTitle: true,
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.numbers, size: 64, color: Colors.blue[700]),
              const SizedBox(height: 24),
              const Text(
                'Your UPI Number',
                style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 12),
              Card(
                elevation: 2,
                child: Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: SelectableText(
                    '9170XXXXXX78@canarabank',
                    style: TextStyle(fontSize: 20, color: Colors.black87, letterSpacing: 1.2),
                  ),
                ),
              ),
              const SizedBox(height: 18),
              ElevatedButton.icon(
                icon: const Icon(Icons.copy),
                label: const Text('Copy UPI Number'),
                onPressed: () {
                  // Add copy logic
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}
