import 'package:flutter/material.dart';

class CardsPage extends StatelessWidget {
  const CardsPage({super.key});

  @override
  Widget build(BuildContext context) {
    final Color canaraBlue = const Color(0xFF0072BC);

    return Scaffold(
      backgroundColor: const Color(0xFFF7F9FB),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: const Text(
          'Cards',
          style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
        iconTheme: IconThemeData(color: canaraBlue),
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 18.0, vertical: 16),
        child: Column(
          children: [
            _cardSection(
              icon: Icons.credit_card,
              title: 'My Debit Cards',
              subtitle: 'View your Debit Cards and Manage services',
              buttonText: 'View Debit Cards',
              buttonColor: canaraBlue,
              imageAsset: 'assets/images/atm-card.png', // Replace with your asset
            ),
            const SizedBox(height: 24),
            _cardSection(
              icon: Icons.credit_card_outlined,
              title: 'My Credit Cards',
              subtitle: 'View your Credit Cards and Manage services',
              buttonText: 'View Credit Cards',
              buttonColor: canaraBlue,
              imageAsset: 'assets/images/atm-card.png', // Replace with your asset
            ),
          ],
        ),
      ),
    );
  }

  Widget _cardSection({
    required IconData icon,
    required String title,
    required String subtitle,
    required String buttonText,
    required Color buttonColor,
    required String imageAsset,
  }) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 4,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        children: [
          CircleAvatar(
            backgroundColor: buttonColor.withOpacity(0.1),
            child: Icon(icon, color: buttonColor, size: 28),
            radius: 28,
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                const SizedBox(height: 4),
                Text(subtitle, style: const TextStyle(color: Colors.black54, fontSize: 13)),
                const SizedBox(height: 12),
                SizedBox(
                  width: 160,
                  height: 36,
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: buttonColor,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(22)),
                    ),
                    onPressed: () {},
                    child: Text(buttonText, style: const TextStyle(color: Colors.white)),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 8),
          Image.asset(
            imageAsset,
            height: 38,
            width: 38,
            fit: BoxFit.contain,
          ),
        ],
      ),
    );
  }
}
