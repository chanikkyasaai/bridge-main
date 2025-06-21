import 'package:canara_ai/screens/nav/tabs/upi/base_upi.dart';
import 'package:flutter/material.dart';

class ApprovePaymentPage extends StatelessWidget {
  final List<Map<String, dynamic>> pendingPayments = [
    {'from': 'John Doe', 'amount': '₹500.00', 'time': '2 minutes ago', 'note': 'Dinner bill split', 'upi': 'john@paytm'},
    {'from': 'Sarah Smith', 'amount': '₹1,200.00', 'time': '5 minutes ago', 'note': 'Rent payment', 'upi': 'sarah@gpay'},
    {'from': 'Coffee Shop', 'amount': '₹250.00', 'time': '10 minutes ago', 'note': 'Morning coffee', 'upi': 'coffee@phonepe'},
  ];

  @override
  Widget build(BuildContext context) {
    return BasePage(
      title: 'Approve Payments',
      child: Container(
        color: Colors.white,
        child: Column(
          children: [
            Container(
              padding: EdgeInsets.all(16),
              child: Row(
                children: [
                  Icon(Icons.pending_actions, color: Colors.orange[600]),
                  SizedBox(width: 8),
                  Text(
                    'Pending Approvals',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                  Spacer(),
                  Chip(
                    label: Text('${pendingPayments.length}'),
                    backgroundColor: Colors.orange[100],
                  ),
                ],
              ),
            ),
            Expanded(
              child: ListView.builder(
                padding: EdgeInsets.symmetric(horizontal: 16),
                itemCount: pendingPayments.length,
                itemBuilder: (context, index) {
                  final payment = pendingPayments[index];
                  return Card(
                    margin: EdgeInsets.only(bottom: 12),
                    child: Padding(
                      padding: EdgeInsets.all(16),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              CircleAvatar(
                                backgroundColor: Colors.orange[100],
                                child: Icon(Icons.person, color: Colors.orange[600]),
                              ),
                              SizedBox(width: 12),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Text(
                                      payment['from'],
                                      style: TextStyle(fontWeight: FontWeight.bold),
                                    ),
                                    Text(
                                      payment['upi'],
                                      style: TextStyle(color: Colors.grey[600], fontSize: 12),
                                    ),
                                  ],
                                ),
                              ),
                              Column(
                                crossAxisAlignment: CrossAxisAlignment.end,
                                children: [
                                  Text(
                                    payment['amount'],
                                    style: TextStyle(
                                      fontSize: 18,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.green[600],
                                    ),
                                  ),
                                  Text(
                                    payment['time'],
                                    style: TextStyle(color: Colors.grey[500], fontSize: 12),
                                  ),
                                ],
                              ),
                            ],
                          ),
                          SizedBox(height: 8),
                          Text(
                            'Note: ${payment['note']}',
                            style: TextStyle(color: Colors.grey[600]),
                          ),
                          SizedBox(height: 12),
                          Row(
                            children: [
                              Expanded(
                                child: OutlinedButton(
                                  onPressed: () => _declinePayment(context, payment),
                                  style: OutlinedButton.styleFrom(
                                    foregroundColor: Colors.red[600],
                                    side: BorderSide(color: Colors.red[600]!),
                                  ),
                                  child: Text('Decline'),
                                ),
                              ),
                              SizedBox(width: 12),
                              Expanded(
                                child: ElevatedButton(
                                  onPressed: () => _approvePayment(context, payment),
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: Colors.green[600],
                                  ),
                                  child: Text('Approve', style: TextStyle(color: Colors.white)),
                                ),
                              ),
                            ],
                          ),
                        ],
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

  void _approvePayment(BuildContext context, Map<String, dynamic> payment) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Approve Payment'),
        content: Text('Approve payment of ${payment['amount']} from ${payment['from']}?'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: Text('Cancel')),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text('Payment approved successfully!'),
                  backgroundColor: Colors.green,
                ),
              );
            },
            child: Text('Confirm'),
          ),
        ],
      ),
    );
  }

  void _declinePayment(BuildContext context, Map<String, dynamic> payment) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('Payment declined'),
        backgroundColor: Colors.red,
      ),
    );
  }
}
