import 'package:flutter/material.dart';

class CreditCardsPage extends StatelessWidget {
  const CreditCardsPage({super.key});

  @override
  Widget build(BuildContext context) {
    final Color canaraBlue = const Color(0xFF0072BC);
    final Color canaraDarkBlue = const Color(0xFF003366);

    return Scaffold(
      backgroundColor: const Color(0xFFF7F9FB),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: const Text(
          'Credit Cards',
          style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
        iconTheme: IconThemeData(color: canaraBlue),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 18.0, vertical: 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('My Credit Cards', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18, color: canaraDarkBlue)),
            const SizedBox(height: 10),
            _cardDetailWidget(
              context,
              cardType: 'Credit',
              cardHolder: 'Dear Customer',
              cardNumber: '3245 12** **** 7895',
              validThru: '**/**',
              cvv: '***',
              cardColor: Colors.orange.shade700,
              bankLogo: 'assets/images/transbiglogo.png',
              chipAsset: 'assets/icons/chip.png',
              cardBgAsset: 'assets/images/orangeworld.jpg', // Replace with your asset
              totalLimit: 30000,
              availableLimit: 15000,
              currentOutstanding: 15000,
              cardStatus: 'Active',
              isCredit: true,
            ),
            const SizedBox(height: 18),
            _cardDetailWidget(
              context,
              cardType: 'Credit',
              cardHolder: 'Dear Customer',
              cardNumber: '5678 34** **** 5678',
              validThru: '**/**',
              cvv: '***',
              cardColor: Colors.deepOrange.shade700,
              bankLogo: 'assets/images/transbiglogo.png',
              chipAsset: 'assets/icons/chip.png',
              cardBgAsset: 'assets/images/blueworld.jpg', // Replace with your asset
              totalLimit: 50000,
              availableLimit: 20000,
              currentOutstanding: 30000,
              cardStatus: 'Active',
              isCredit: true,
            ),
          ],
        ),
      ),
    );
  }

  Widget _cardDetailWidget(
    BuildContext context, {
    required String cardType,
    required String cardHolder,
    required String cardNumber,
    required String validThru,
    required String cvv,
    required Color cardColor,
    required String bankLogo,
    required String chipAsset,
    required String cardBgAsset,
    required double totalLimit,
    required double availableLimit,
    required double currentOutstanding,
    required String cardStatus,
    bool isCredit = false,
  }) {
    final Color canaraBlue = const Color(0xFF0072BC);

    return Container(
      margin: const EdgeInsets.only(bottom: 6),
      padding: const EdgeInsets.all(16),
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
      child: Column(
        children: [
          // Card Visual
          Container(
            width: double.infinity,
            height: 160,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(16),
              gradient: LinearGradient(
                colors: [cardColor, cardColor.withOpacity(0.8)],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              image: DecorationImage(
                image: AssetImage(cardBgAsset),
                fit: BoxFit.cover,
                opacity: 0.18,
              ),
            ),
            child: Stack(
              children: [
                Positioned(
                  top: 16,
                  left: 18,
                  child: Image.asset(bankLogo, height: 28),
                ),
                Positioned(
                  top: 16,
                  right: 18,
                  child: Text(
                    cardType == 'Credit' ? 'Platinum Credit Card' : 'Platinum Debit Card',
                    style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 13),
                  ),
                ),
                Positioned(
                  left: 18,
                  top: 48,
                  child: Image.asset(chipAsset, height: 24),
                ),
                Positioned(
                  left: 18,
                  top: 78,
                  child: Text(
                    cardNumber,
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 16,
                      letterSpacing: 2,
                    ),
                  ),
                ),
                Positioned(
                  left: 18,
                  bottom: 14,
                  child: Text(
                    'VALID TO: $validThru',
                    style: const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                ),
                Positioned(
                  right: 18,
                  bottom: 14,
                  child: Text(
                    'CVV: $cvv',
                    style: const TextStyle(color: Colors.white70, fontSize: 11),
                  ),
                ),
                Positioned(
                  right: 18,
                  top: 48,
                  child: Text(
                    cardHolder,
                    style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w500, fontSize: 13),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),
          // Card Details
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _cardInfoTile('Total Limit', '₹ ${totalLimit.toStringAsFixed(2)}', canaraBlue),
              _cardInfoTile('Available Limit', '₹ ${availableLimit.toStringAsFixed(2)}', canaraBlue),
              _cardInfoTile('Current Outstanding', '₹ ${currentOutstanding.toStringAsFixed(2)}', canaraBlue),
              _cardInfoTile('Status', cardStatus, Colors.green),
            ],
          ),
          if (isCredit) ...[
            const SizedBox(height: 12),
            // Latest Bill Summary for Credit Card
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.blue.shade50,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Text('Latest Bill Summary', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15)),
                      const Spacer(),
                      TextButton(
                        onPressed: () {},
                        child: const Text('View Bill', style: TextStyle(fontSize: 13)),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  Row(
                    children: [
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: const [
                            Text('Total Billed Amount', style: TextStyle(fontSize: 12)),
                            SizedBox(height: 2),
                            Text('₹ 15,000.00', style: TextStyle(fontWeight: FontWeight.bold, color: Colors.black87)),
                            SizedBox(height: 8),
                            Text('Minimum Due', style: TextStyle(fontSize: 12)),
                            SizedBox(height: 2),
                            Text('₹ 1,500.00', style: TextStyle(fontWeight: FontWeight.bold, color: Colors.black87)),
                            SizedBox(height: 8),
                            Text('Reward Points', style: TextStyle(fontSize: 12)),
                            SizedBox(height: 2),
                            Text('1500', style: TextStyle(fontWeight: FontWeight.bold, color: Colors.black87)),
                          ],
                        ),
                      ),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text('Payment Due Date', style: TextStyle(fontSize: 12)),
                            const SizedBox(height: 2),
                            Container(
                              padding: const EdgeInsets.all(8),
                              decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(8),
                                border: Border.all(color: Colors.blue.shade100),
                              ),
                              child: Column(
                                children: const [
                                  Icon(Icons.calendar_month, color: Colors.blue, size: 22),
                                  SizedBox(height: 4),
                                  Text('25 April', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 13)),
                                  SizedBox(height: 2),
                                  Text('Due in 15 Days', style: TextStyle(fontSize: 11, color: Colors.black54)),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 10),
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: canaraBlue,
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(22)),
                        padding: const EdgeInsets.symmetric(vertical: 10),
                      ),
                      onPressed: () {},
                      child: const Text('Pay Now', style: TextStyle(color: Colors.white)),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _cardInfoTile(String label, String value, Color color) {
    return Expanded(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: const TextStyle(fontSize: 12, color: Colors.black54)),
          const SizedBox(height: 2),
          Text(
            value,
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 13, color: color),
          ),
        ],
      ),
    );
  }
}
