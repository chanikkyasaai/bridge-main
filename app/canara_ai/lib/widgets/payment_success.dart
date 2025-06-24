import 'package:flutter/material.dart';
import 'dart:async';

class PaymentSuccessPage extends StatefulWidget {
  final String recipientName;
  final double amount;

  const PaymentSuccessPage({
    super.key,
    required this.recipientName,
    required this.amount,
  });

  @override
  State<PaymentSuccessPage> createState() => _PaymentSuccessPageState();
}

class _PaymentSuccessPageState extends State<PaymentSuccessPage> with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _scaleAnimation;

  final Color canaraBlue = const Color(0xFF0072BC);
  final Color canaraGreen = const Color(0xFF00C853);

  @override
  void initState() {
    super.initState();

    // Animation setup
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 700),
    );

    _scaleAnimation = CurvedAnimation(parent: _controller, curve: Curves.elasticOut);

    _controller.forward();

    // Auto exit after 5 seconds
    Timer(const Duration(seconds: 5), () {
      Navigator.of(context).pop(); // You can change this to push to HomePage
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  String formatAmount(double amount) {
    return '₹${amount.toStringAsFixed(2)}';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: ScaleTransition(
          scale: _scaleAnimation,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 30, vertical: 40),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(20),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.05),
                  blurRadius: 20,
                  offset: const Offset(0, 10),
                ),
              ],
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.check_circle_rounded, size: 80, color: canaraGreen),
                const SizedBox(height: 20),
                Text(
                  'Payment Successful',
                  style: TextStyle(
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    color: canaraBlue,
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  'To: ${widget.recipientName}',
                  style: const TextStyle(fontSize: 16),
                ),
                const SizedBox(height: 8),
                Text(
                  formatAmount(widget.amount),
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    color: canaraGreen,
                  ),
                ),
                const SizedBox(height: 20),
                const CircularProgressIndicator(),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
