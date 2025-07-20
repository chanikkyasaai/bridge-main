import 'package:flutter/material.dart';
import 'package:canara_ai/logging/behaviour_route_tracker.dart';
import 'package:canara_ai/logging/log_touch_data.dart';
import 'package:canara_ai/logging/logger_instance.dart';
import 'package:canara_ai/main.dart';
import 'package:flutter/material.dart';

class StatementPage extends StatefulWidget {
  const StatementPage({super.key});

  @override
  State<StatementPage> createState() => _StatementPageState();
}

class _StatementPageState extends State<StatementPage> {

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
    final Color canaraBlue = const Color(0xFF0072BC);

    return Scaffold(
      backgroundColor: const Color(0xFFF7F9FB),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: const Text('Account Statement', style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold)),
        iconTheme: IconThemeData(color: canaraBlue),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 4, offset: const Offset(0, 2))],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Account: 110084150765', style: TextStyle(fontWeight: FontWeight.bold, color: canaraBlue)),
                  const SizedBox(height: 8),
                  const Text('Statement Period: Last 30 days'),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: () {},
                          icon: const Icon(Icons.download),
                          label: const Text('Download PDF'),
                          style: ElevatedButton.styleFrom(backgroundColor: canaraBlue, foregroundColor: Colors.white),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: () {},
                          icon: const Icon(Icons.email),
                          label: const Text('Email'),
                          style: OutlinedButton.styleFrom(foregroundColor: canaraBlue),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),
            Expanded(
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                  boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.04), blurRadius: 4, offset: const Offset(0, 2))],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('Recent Transactions', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: canaraBlue)),
                    const Divider(height: 24),
                    Expanded(
                      child: ListView(
                        children: [
                          _transactionTile('UPI Payment', 'To Ravi Kumar', '-₹5,000', '22 Jun 2025', Colors.red),
                          _transactionTile('Salary Credit', 'Monthly Salary', '+₹45,000', '20 Jun 2025', Colors.green),
                          _transactionTile('Bill Payment', 'Electricity Bill', '-₹2,500', '18 Jun 2025', Colors.red),
                          _transactionTile('ATM Withdrawal', 'Cash Withdrawal', '-₹10,000', '15 Jun 2025', Colors.red),
                          _transactionTile('Interest Credit', 'Savings Interest', '+₹125', '10 Jun 2025', Colors.green),
                        ],
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

  Widget _transactionTile(String title, String subtitle, String amount, String date, Color amountColor) {
    return ListTile(
      contentPadding: const EdgeInsets.symmetric(vertical: 4),
      title: Text(title, style: const TextStyle(fontWeight: FontWeight.w500)),
      subtitle: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(subtitle),
          const SizedBox(height: 2),
          Text(date, style: TextStyle(fontSize: 12, color: Colors.grey[600])),
        ],
      ),
      trailing: Text(amount, style: TextStyle(fontWeight: FontWeight.bold, color: amountColor)),
    );
  }
}
