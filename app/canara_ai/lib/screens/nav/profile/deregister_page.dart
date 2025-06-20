import 'package:flutter/material.dart';

class DeRegisterPage extends StatelessWidget {
  const DeRegisterPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('De-Register', style: TextStyle(fontWeight: FontWeight.bold)),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.logout, size: 64, color: Colors.red[700]),
            const SizedBox(height: 24),
            const Text(
              'De-Register Account',
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 12),
            const Text(
              'Are you sure you want to de-register from this app? This action cannot be undone.',
              style: TextStyle(fontSize: 16, color: Colors.grey),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 32),
            ElevatedButton.icon(
              icon: const Icon(Icons.warning_amber_rounded),
              label: const Text('De-Register'),
              style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
              onPressed: () {
                // Add your de-register logic here
              },
            ),
          ],
        ),
      ),
    );
  }
}
