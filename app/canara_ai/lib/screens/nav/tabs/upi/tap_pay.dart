import 'package:canara_ai/screens/nav/tabs/upi/base_upi.dart';
import 'package:flutter/material.dart';

import 'package:canara_ai/main.dart';
import 'package:canara_ai/logging/behaviour_route_tracker.dart';
import 'package:canara_ai/logging/log_touch_data.dart';
import 'package:canara_ai/logging/logger_instance.dart';

class TapPayPage extends StatefulWidget {
  @override
  _TapPayPageState createState() => _TapPayPageState();
}

class _TapPayPageState extends State<TapPayPage> {
  bool _isNFCEnabled = false;
  bool _isSearching = false;

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
      title: 'Tap & Pay',
      child: Container(
        color: Colors.white,
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Column(
            children: [
              // NFC Animation
              Container(
                height: 250,
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    // Outer ripple
                    AnimatedContainer(
                      duration: Duration(seconds: 2),
                      width: _isSearching ? 200 : 150,
                      height: _isSearching ? 200 : 150,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(
                          color: Colors.blue[300]!.withOpacity(_isSearching ? 0.3 : 0.6),
                          width: 2,
                        ),
                      ),
                    ),
                    // Middle ripple
                    AnimatedContainer(
                      duration: Duration(milliseconds: 1500),
                      width: _isSearching ? 150 : 100,
                      height: _isSearching ? 150 : 100,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        border: Border.all(
                          color: Colors.blue[400]!.withOpacity(_isSearching ? 0.5 : 0.8),
                          width: 2,
                        ),
                      ),
                    ),
                    // Inner circle with NFC icon
                    Container(
                      width: 80,
                      height: 80,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.blue[600],
                        boxShadow: [
                          BoxShadow(
                            color: Colors.blue[600]!.withOpacity(0.3),
                            blurRadius: 10,
                            spreadRadius: 2,
                          ),
                        ],
                      ),
                      child: Icon(
                        Icons.nfc,
                        size: 40,
                        color: Colors.white,
                      ),
                    ),
                  ],
                ),
              ),
              SizedBox(height: 32),

              Text(
                _isSearching ? 'Searching for devices...' : 'Tap & Pay with NFC',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Colors.black87,
                ),
                textAlign: TextAlign.center,
              ),
              SizedBox(height: 16),

              Text(
                _isSearching ? 'Hold your phone near the payment terminal' : 'Enable NFC to make contactless payments',
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.grey[600],
                ),
                textAlign: TextAlign.center,
              ),
              SizedBox(height: 32),

              // NFC Status Card
              Container(
                width: double.infinity,
                padding: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: _isNFCEnabled ? Colors.green[50] : Colors.orange[50],
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: _isNFCEnabled ? Colors.green[200]! : Colors.orange[200]!,
                  ),
                ),
                child: Row(
                  children: [
                    Icon(
                      _isNFCEnabled ? Icons.check_circle : Icons.warning,
                      color: _isNFCEnabled ? Colors.green[600] : Colors.orange[600],
                    ),
                    SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            _isNFCEnabled ? 'NFC Enabled' : 'NFC Disabled',
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              color: _isNFCEnabled ? Colors.green[800] : Colors.orange[800],
                            ),
                          ),
                          Text(
                            _isNFCEnabled ? 'Ready for contactless payments' : 'Enable NFC in device settings',
                            style: TextStyle(
                              color: _isNFCEnabled ? Colors.green[700] : Colors.orange[700],
                            ),
                          ),
                        ],
                      ),
                    ),
                    Switch(
                      value: _isNFCEnabled,
                      onChanged: (value) {
                        setState(() {
                          _isNFCEnabled = value;
                          if (value) {
                            _startSearching();
                          } else {
                            _isSearching = false;
                          }
                        });
                      },
                      activeColor: Colors.blue[600],
                    ),
                  ],
                ),
              ),
              SizedBox(height: 24),

              // Payment Limits
              Container(
                padding: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.grey[50],
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Payment Limits',
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                    SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text('Per Transaction:'),
                        Text('₹5,000', style: TextStyle(fontWeight: FontWeight.w500)),
                      ],
                    ),
                    SizedBox(height: 4),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text('Daily Limit:'),
                        Text('₹25,000', style: TextStyle(fontWeight: FontWeight.w500)),
                      ],
                    ),
                  ],
                ),
              ),

              Spacer(),

              if (_isNFCEnabled && !_isSearching)
                SizedBox(
                  width: double.infinity,
                  height: 50,
                  child: ElevatedButton(
                    onPressed: () => _startSearching(),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue[600],
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(25)),
                    ),
                    child: Text('Start Payment', style: TextStyle(fontSize: 16, color: Colors.white)),
                  ),
                ),

              if (_isSearching)
                SizedBox(
                  width: double.infinity,
                  height: 50,
                  child: OutlinedButton(
                    onPressed: () => setState(() => _isSearching = false),
                    style: OutlinedButton.styleFrom(
                      side: BorderSide(color: Colors.red[600]!),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(25)),
                    ),
                    child: Text('Cancel', style: TextStyle(fontSize: 16, color: Colors.red[600])),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  void _startSearching() {
    setState(() => _isSearching = true);

    // Simulate payment detection after 3 seconds
    Future.delayed(Duration(seconds: 3), () {
      if (_isSearching) {
        _simulatePayment();
      }
    });
  }

  void _simulatePayment() {
    setState(() => _isSearching = false);
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Payment Terminal Detected'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text('Merchant: Electronics Store'),
            Text('Amount: ₹2,499.00'),
            SizedBox(height: 16),
            Text('Confirm payment?'),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: Text('Cancel')),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text('Payment successful!'),
                  backgroundColor: Colors.green,
                ),
              );
            },
            child: Text('Pay Now'),
          ),
        ],
      ),
    );
  }
}
