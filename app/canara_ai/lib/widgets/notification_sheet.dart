import 'package:flutter/material.dart';

class NotificationSheet extends StatelessWidget {
  final Color canaraBlue;

  const NotificationSheet({super.key, required this.canaraBlue});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 350,
      child: Column(
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
            child: Row(
              children: [
                const Text(
                  'Notification',
                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18),
                ),
                const Spacer(),
                IconButton(
                  icon: Icon(Icons.close, color: canaraBlue),
                  onPressed: () => Navigator.pop(context),
                ),
              ],
            ),
          ),
          Container(
            width: double.infinity,
            alignment: Alignment.centerLeft,
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 8),
            child: Text(
              'MBS',
              style: TextStyle(
                color: canaraBlue,
                fontWeight: FontWeight.bold,
                fontSize: 15,
              ),
            ),
          ),
          Divider(color: canaraBlue, thickness: 2, height: 0),
          const SizedBox(height: 40),
          const Center(
            child: Text(
              'No record found!',
              style: TextStyle(fontSize: 16, color: Colors.black54),
            ),
          ),
        ],
      ),
    );
  }
}
