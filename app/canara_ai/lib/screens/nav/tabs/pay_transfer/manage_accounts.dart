import 'package:flutter/material.dart';

class HistoryPage extends StatelessWidget {
  const HistoryPage({Key? key}) : super(key: key);

  final Color canaraBlue = const Color(0xFF1976D2);
  final Color canaraDarkBlue = const Color(0xFF0D47A1);

  final List<Map<String, dynamic>> _transactions = const [
    {
      'type': 'Transfer',
      'amount': '5000',
      'date': '21 Jun 2025',
      'status': 'Success',
      'icon': Icons.send,
      'isDebit': true,
    },
    {
      'type': 'UPI Payment',
      'amount': '250',
      'date': '20 Jun 2025',
      'status': 'Success',
      'icon': Icons.payment,
      'isDebit': true,
    },
    {
      'type': 'Salary Credit',
      'amount': '50000',
      'date': '19 Jun 2025',
      'status': 'Success',
      'icon': Icons.account_balance_wallet,
      'isDebit': false,
    },
    {
      'type': 'ATM Withdrawal',
      'amount': '2000',
      'date': '18 Jun 2025',
      'status': 'Success',
      'icon': Icons.atm,
      'isDebit': true,
    },
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: false,
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        backgroundColor: Colors.grey[100],
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.black),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text(
          'Transaction History',
          style: TextStyle(
            color: Colors.black,
            fontSize: 18,
            fontWeight: FontWeight.w500,
          ),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // Filter Card
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.1),
                    blurRadius: 10,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: Row(
                children: [
                  Icon(Icons.filter_list, color: canaraBlue),
                  const SizedBox(width: 12),
                  const Text(
                    'Last 30 Days',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  const Spacer(),
                  TextButton(
                    onPressed: () {},
                    child: const Text('Filter'),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
            // Transaction List
            Expanded(
              child: ListView.builder(
                itemCount: _transactions.length,
                itemBuilder: (context, index) {
                  final transaction = _transactions[index];
                  return Container(
                    margin: const EdgeInsets.only(bottom: 12),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.1),
                          blurRadius: 10,
                          offset: const Offset(0, 2),
                        ),
                      ],
                    ),
                    child: ListTile(
                      contentPadding: const EdgeInsets.all(16),
                      leading: CircleAvatar(
                        backgroundColor: transaction['isDebit'] ? Colors.red.withOpacity(0.1) : Colors.green.withOpacity(0.1),
                        child: Icon(
                          transaction['icon'],
                          color: transaction['isDebit'] ? Colors.red : Colors.green,
                        ),
                      ),
                      title: Text(
                        transaction['type'],
                        style: const TextStyle(
                          fontWeight: FontWeight.w600,
                          fontSize: 16,
                        ),
                      ),
                      subtitle: Text(
                        '${transaction['date']} • ${transaction['status']}',
                        style: const TextStyle(
                          color: Colors.grey,
                          fontSize: 14,
                        ),
                      ),
                      trailing: Text(
                        '${transaction['isDebit'] ? '-' : '+'}₹${transaction['amount']}',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                          color: transaction['isDebit'] ? Colors.red : Colors.green,
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
