import 'package:canara_ai/logging/button_wrapper.dart';
import 'package:canara_ai/logging/typing_tracker.dart';
import 'package:canara_ai/screens/nav/tabs/upi/base_upi.dart';
import 'package:flutter/material.dart';

import '../../../../main.dart';
import 'package:canara_ai/logging/behaviour_route_tracker.dart';
import 'package:canara_ai/logging/log_touch_data.dart';
import 'package:canara_ai/logging/logger_instance.dart';

class PayContactPage extends StatefulWidget {
  @override
  _PayContactPageState createState() => _PayContactPageState();
}

class _PayContactPageState extends State<PayContactPage> {
  final _searchController = TextEditingController();

  final List<Map<String, String>> contacts = [
    {'name': 'John Doe', 'phone': '+91 9876543210', 'upi': 'john@paytm'},
    {'name': 'Sarah Smith', 'phone': '+91 8765432109', 'upi': 'sarah@gpay'},
    {'name': 'Mike Johnson', 'phone': '+91 7654321098', 'upi': 'mike@phonepe'},
    {'name': 'Emma Wilson', 'phone': '+91 6543210987', 'upi': 'emma@paytm'},
  ];

  late BehaviorLogger logger;
  late BehaviorRouteTracker tracker;
  bool _subscribed = false;

  
@override
  Widget build(BuildContext context) {
    return BasePage(
      title: 'Pay to Contact',
      child: Column(
        children: [
          Container(
            color: Colors.white,
            padding: EdgeInsets.all(16),
            child: TypingFieldTracker(
              controller: _searchController,
              fieldName: 'search_contact',
              screenName: 'Pay_Contact',
              logger: logger,
              child: TextField(
                controller: _searchController,
                decoration: InputDecoration(
                  hintText: 'Search contacts or enter mobile number',
                  prefixIcon: Icon(Icons.search),
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                ),
                onChanged: (value) => setState(() {}),
              ),
            ),
          ),
          Expanded(
            child: Container(
              color: Colors.white,
              child: ListView.builder(
                padding: EdgeInsets.symmetric(horizontal: 16),
                itemCount: contacts.length,
                itemBuilder: (context, index) {
                  final contact = contacts[index];
                  return Card(
                    margin: EdgeInsets.only(bottom: 8),
                    child: ListTile(
                      leading: CircleAvatar(
                        backgroundColor: Colors.blue[100],
                        child: Text(
                          contact['name']![0],
                          style: TextStyle(color: Colors.blue[600], fontWeight: FontWeight.bold),
                        ),
                      ),
                      title: Text(contact['name']!),
                      subtitle: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(contact['phone']!),
                          Text(contact['upi']!, style: TextStyle(color: Colors.green[600])),
                        ],
                      ),
                      trailing: LoggedButton(
                        eventName: 'button_press',
                        logger: logger,
                        eventData: {
                          'button_name': 'button_pay',
                          'screen': 'Pay_Contact',
                        },
                        onTap: () => _showPaymentDialog(context, contact),
                        child: ElevatedButton(
                          onPressed: null, // Disabled, handled by LoggedButton
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.blue[600],
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                          ),
                          child: Text('Pay', style: TextStyle(color: Colors.white)),
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
          ),
        ],
      ),
    );
  }

  void _showPaymentDialog(BuildContext context, Map<String, String> contact) {
    final amountController = TextEditingController();
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Pay ${contact['name']}'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            TypingFieldTracker(
              controller: amountController,
              fieldName: 'amount_field',
              screenName: 'Pay_Contact',
              logger: logger,
              child: TextField(
                controller: amountController,
                keyboardType: TextInputType.number,
                decoration: InputDecoration(
                  labelText: 'Amount',
                  prefixText: '₹ ',
                  border: OutlineInputBorder(),
                ),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: Text('Cancel')),
          LoggedButton(
            eventName: 'send_payment_button',
            logger: logger,
            eventData: {
              'contact_name': contact['name'],
              'amount': amountController.text,
            },
            onTap: () {
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text('Payment sent to ${contact['name']}')),
              );
            },
            child: ElevatedButton(
              onPressed: null, // Disabled, handled by LoggedButton
              child: Text('Send'),
            ),
          ),
        ],
      ),
    );
  }
}
