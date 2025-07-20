import 'package:canara_ai/screens/nav/tabs/upi/base_upi.dart';
import 'package:flutter/material.dart';

import 'package:canara_ai/main.dart';
import 'package:canara_ai/logging/behaviour_route_tracker.dart';
import 'package:canara_ai/logging/log_touch_data.dart';
import 'package:canara_ai/logging/logger_instance.dart';

class UPILitePage extends StatefulWidget {
  @override
  _UPILitePageState createState() => _UPILitePageState();
}

class _UPILitePageState extends State<UPILitePage> {
  double _liteBalance = 500.00;
  bool _isLiteEnabled = true;

  final List<Map<String, dynamic>> _liteTransactions = [
    {
      'type': 'sent',
      'amount': 50.0,
      'to': 'Coffee Shop',
      'time': '10:30 AM',
      'date': 'Today',
    },
    {
      'type': 'received',
      'amount': 100.0,
      'from': 'John Doe',
      'time': '09:15 AM',
      'date': 'Today',
    },
    {
      'type': 'sent',
      'amount': 25.0,
      'to': 'Metro Card',
      'time': '08:45 AM',
      'date': 'Today',
    },
  ];

  late BehaviorLogger logger;
  late BehaviorRouteTracker tracker;
  bool _subscribed = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    if (!_subscribed) {
      final route = ModalRoute.of(context);
      if (route is PageRoute) {
        tracker = BehaviorRouteTracker(logger, context);
        routeObserver.subscribe(tracker, route);
        _subscribed = true;
      }
    }
  }

  @override
  void initState() {
    super.initState();
    logger = AppLogger.logger;
  }

  @override
  void dispose() {
    routeObserver.unsubscribe(tracker);
  }

  @override
  Widget build(BuildContext context) {
    return BasePage(
      title: 'UPI Lite',
      child: Container(
        color: Colors.white,
        child: Column(
          children: [
            // UPI Lite Balance Card
            Container(
              margin: EdgeInsets.all(16),
              padding: EdgeInsets.all(20),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [Colors.green[400]!, Colors.green[600]!],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        'UPI Lite Balance',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                      Container(
                        padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.2),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          _isLiteEnabled ? 'ACTIVE' : 'INACTIVE',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ],
                  ),
                  SizedBox(height: 16),
                  Text(
                    '₹${_liteBalance.toStringAsFixed(2)}',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 32,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 8),
                  Text(
                    'Available for instant payments up to ₹200',
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.9),
                      fontSize: 14,
                    ),
                  ),
                ],
              ),
            ),

            // Quick Actions
            Container(
              padding: EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                children: [
                  Expanded(
                    child: _quickActionButton(
                      icon: Icons.add,
                      label: 'Add Money',
                      onTap: () => _showAddMoneyDialog(),
                    ),
                  ),
                  SizedBox(width: 12),
                  Expanded(
                    child: _quickActionButton(
                      icon: Icons.send,
                      label: 'Send Money',
                      onTap: () => _showSendMoneyDialog(),
                    ),
                  ),
                  SizedBox(width: 12),
                  Expanded(
                    child: _quickActionButton(
                      icon: Icons.settings,
                      label: 'Settings',
                      onTap: () => _showSettingsDialog(),
                    ),
                  ),
                ],
              ),
            ),
            SizedBox(height: 24),

            // Features Section
            Container(
              padding: EdgeInsets.symmetric(horizontal: 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'UPI Lite Features',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  SizedBox(height: 16),
                  _featureItem(
                    Icons.flash_on,
                    'Instant Payments',
                    'Make payments up to ₹200 without PIN',
                  ),
                  _featureItem(
                    Icons.offline_bolt,
                    'Offline Payments',
                    'Works even with poor network connectivity',
                  ),
                  _featureItem(
                    Icons.security,
                    'Secure & Safe',
                    'Pre-loaded balance with transaction limits',
                  ),
                ],
              ),
            ),
            SizedBox(height: 24),

            // Recent Transactions
            Expanded(
              child: Container(
                padding: EdgeInsets.symmetric(horizontal: 16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Recent UPI Lite Transactions',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    SizedBox(height: 16),
                    Expanded(
                      child: ListView.builder(
                        itemCount: _liteTransactions.length,
                        itemBuilder: (context, index) {
                          final transaction = _liteTransactions[index];
                          return _transactionItem(transaction);
                        },
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _quickActionButton({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: EdgeInsets.symmetric(vertical: 16),
        decoration: BoxDecoration(
          color: Colors.blue[50],
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.blue[200]!),
        ),
        child: Column(
          children: [
            Icon(icon, color: Colors.blue[600], size: 24),
            SizedBox(height: 8),
            Text(
              label,
              style: TextStyle(
                color: Colors.blue[800],
                fontSize: 12,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _featureItem(IconData icon, String title, String description) {
    return Padding(
      padding: EdgeInsets.only(bottom: 16),
      child: Row(
        children: [
          Container(
            padding: EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.green[100],
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(icon, color: Colors.green[600], size: 20),
          ),
          SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 14,
                  ),
                ),
                Text(
                  description,
                  style: TextStyle(
                    color: Colors.grey[600],
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _transactionItem(Map<String, dynamic> transaction) {
    bool isSent = transaction['type'] == 'sent';
    return Container(
      margin: EdgeInsets.only(bottom: 8),
      padding: EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.grey[50],
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          Container(
            padding: EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: isSent ? Colors.red[100] : Colors.green[100],
              borderRadius: BorderRadius.circular(20),
            ),
            child: Icon(
              isSent ? Icons.arrow_upward : Icons.arrow_downward,
              color: isSent ? Colors.red[600] : Colors.green[600],
              size: 16,
            ),
          ),
          SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  isSent ? 'To ${transaction['to']}' : 'From ${transaction['from']}',
                  style: TextStyle(fontWeight: FontWeight.w500),
                ),
                Text(
                  '${transaction['date']} • ${transaction['time']}',
                  style: TextStyle(color: Colors.grey[600], fontSize: 12),
                ),
              ],
            ),
          ),
          Text(
            '${isSent ? '-' : '+'}₹${transaction['amount'].toStringAsFixed(0)}',
            style: TextStyle(
              color: isSent ? Colors.red[600] : Colors.green[600],
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  void _showAddMoneyDialog() {
    final amountController = TextEditingController();
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Add Money to UPI Lite'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: amountController,
              keyboardType: TextInputType.number,
              decoration: InputDecoration(
                labelText: 'Amount (Max ₹2000)',
                prefixText: '₹ ',
                border: OutlineInputBorder(),
              ),
            ),
            SizedBox(height: 8),
            Text(
              'Current Balance: ₹${_liteBalance.toStringAsFixed(2)}',
              style: TextStyle(color: Colors.grey[600]),
            ),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: Text('Cancel')),
          ElevatedButton(
            onPressed: () {
              double amount = double.tryParse(amountController.text) ?? 0;
              if (amount > 0 && amount <= 2000) {
                setState(() {
                  _liteBalance += amount;
                });
                Navigator.pop(context);
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('₹${amount.toStringAsFixed(0)} added to UPI Lite')),
                );
              }
            },
            child: Text('Add Money'),
          ),
        ],
      ),
    );
  }

  void _showSendMoneyDialog() {
    final amountController = TextEditingController();
    final upiController = TextEditingController();
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Send Money via UPI Lite'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TextField(
              controller: upiController,
              decoration: InputDecoration(
                labelText: 'UPI ID',
                border: OutlineInputBorder(),
              ),
            ),
            SizedBox(height: 12),
            TextField(
              controller: amountController,
              keyboardType: TextInputType.number,
              decoration: InputDecoration(
                labelText: 'Amount (Max ₹200)',
                prefixText: '₹ ',
                border: OutlineInputBorder(),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: Text('Cancel')),
          ElevatedButton(
            onPressed: () {
              double amount = double.tryParse(amountController.text) ?? 0;
              if (amount > 0 && amount <= 200 && amount <= _liteBalance) {
                setState(() {
                  _liteBalance -= amount;
                });
                Navigator.pop(context);
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('₹${amount.toStringAsFixed(0)} sent successfully')),
                );
              }
            },
            child: Text('Send'),
          ),
        ],
      ),
    );
  }

  void _showSettingsDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('UPI Lite Settings'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            SwitchListTile(
              title: Text('Enable UPI Lite'),
              subtitle: Text('Allow instant payments without PIN'),
              value: _isLiteEnabled,
              onChanged: (value) {
                setState(() => _isLiteEnabled = value);
                Navigator.pop(context);
              },
            ),
            ListTile(
              title: Text('Transaction Limit'),
              subtitle: Text('₹200 per transaction'),
              trailing: Icon(Icons.arrow_forward_ios),
            ),
            ListTile(
              title: Text('Daily Limit'),
              subtitle: Text('₹2000 per day'),
              trailing: Icon(Icons.arrow_forward_ios),
            ),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: Text('Close')),
        ],
      ),
    );
  }
}
