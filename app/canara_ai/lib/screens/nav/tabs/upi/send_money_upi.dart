import 'package:canara_ai/logging/button_wrapper.dart';
import 'package:canara_ai/logging/typing_tracker.dart';
import 'package:canara_ai/screens/nav/tabs/upi/base_upi.dart';
import 'package:flutter/material.dart';

import 'package:canara_ai/main.dart';
import 'package:canara_ai/logging/behaviour_route_tracker.dart';
import 'package:canara_ai/logging/log_touch_data.dart';
import 'package:canara_ai/logging/logger_instance.dart';

class SendMoneyPageUPI extends StatefulWidget {
  @override
  _SendMoneyPageState createState() => _SendMoneyPageState();
}

class _SendMoneyPageState extends State<SendMoneyPageUPI> {
  final _amountController = TextEditingController();
  final _upiController = TextEditingController();
  final _noteController = TextEditingController();

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
      title: 'Send Money',
      child: Container(
        color: Colors.white,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Send money to any UPI app',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 24),

              // --- UPI ID ---
              TypingFieldTracker(
                controller: _upiController,
                fieldName: 'upi_id',
                logger: logger,
                child: TextFormField(
                  controller: _upiController,
                  decoration: InputDecoration(
                    labelText: 'Enter UPI ID',
                    hintText: 'example@paytm',
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                    prefixIcon: const Icon(Icons.alternate_email),
                  ),
                ),
              ),
              const SizedBox(height: 16),

              // --- Amount ---
              TypingFieldTracker(
                controller: _amountController,
                fieldName: 'amount',
                logger: logger,
                child: TextFormField(
                  controller: _amountController,
                  keyboardType: TextInputType.number,
                  decoration: InputDecoration(
                    labelText: 'Amount',
                    prefixText: '₹ ',
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                    prefixIcon: const Icon(Icons.currency_rupee),
                  ),
                ),
              ),
              const SizedBox(height: 16),

              // --- Note ---
              TypingFieldTracker(
                controller: _noteController,
                fieldName: 'note',
                logger: logger,
                child: TextFormField(
                  controller: _noteController,
                  maxLines: 2,
                  decoration: InputDecoration(
                    labelText: 'Note (Optional)',
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                    prefixIcon: const Icon(Icons.note),
                  ),
                ),
              ),
              const SizedBox(height: 24),

              // --- Info Box ---
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.blue[50],
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  children: [
                    Icon(Icons.info, color: Colors.blue[600]),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Money will be sent instantly to the recipient\'s bank account',
                        style: TextStyle(color: Colors.blue[800]),
                      ),
                    ),
                  ],
                ),
              ),
              const Spacer(),

              // --- Send Button with logging ---
              SizedBox(
                width: double.infinity,
                height: 50,
                child: LoggedButton(
                  logger: logger,
                  eventName: 'button_press',
                  eventData: {
                    'button_name': 'send_money',
                    'new_state': 'enabled',
                    'screen': 'Send Money UPI',
                  },
                  onLongPress: () => _showPaymentConfirmation(context),
                  onTap: () => _showPaymentConfirmation(context),
                  child: ElevatedButton(
                    onPressed: null, // Disabled because tap is handled by LoggedButton
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue[600],
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(25)),
                    ),
                    child: const Text('Send Money', style: TextStyle(fontSize: 16, color: Colors.white)),
                  ),
                  onDoubleTap: () => _showPaymentConfirmation(context),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }


  void _showPaymentConfirmation(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Confirm Payment'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('To: ${_upiController.text}'),
            Text('Amount: ₹${_amountController.text}'),
            if (_noteController.text.isNotEmpty) Text('Note: ${_noteController.text}'),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: Text('Cancel')),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              _showSuccessMessage(context);
            },
            child: Text('Confirm'),
          ),
        ],
      ),
    );
  }

  void _showSuccessMessage(BuildContext context) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('Payment sent successfully!'),
        backgroundColor: Colors.green,
      ),
    );
  }
}
