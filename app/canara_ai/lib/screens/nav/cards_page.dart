import 'package:canara_ai/logging/behaviour_route_tracker.dart';
import 'package:canara_ai/logging/log_touch_data.dart';
import 'package:canara_ai/logging/logger_instance.dart';
import 'package:canara_ai/main.dart';
import 'package:flutter/material.dart';

import 'package:canara_ai/screens/nav/cards/debit_card.dart';
import 'package:canara_ai/screens/nav/cards/credit_card.dart';

class CardsPage extends StatefulWidget {
  const CardsPage({super.key});

  @override
  State<CardsPage> createState() => _CardsPageState();
}

class _CardsPageState extends State<CardsPage> {
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
              context: context,
              icon: Icons.credit_card,
              title: 'My Debit Cards',
              subtitle: 'View your Debit Cards and Manage services',
              buttonText: 'View Debit Cards',
              buttonColor: canaraBlue,
              imageAsset: 'assets/images/atm-card.png', // Replace with your asset
                onPressed: () {
                Navigator.of(context).pushNamed(
                  '/debitcards',
                );
                },
              ),
              const SizedBox(height: 24),
              _cardSection(
                context: context,
                icon: Icons.credit_card_outlined,
                title: 'My Credit Cards',
                subtitle: 'View your Credit Cards and Manage services',
                buttonText: 'View Credit Cards',
                buttonColor: canaraBlue,
                imageAsset: 'assets/images/atm-card.png', // Replace with your asset
                onPressed: () {
                Navigator.of(context).pushNamed(
                  '/creditcards',
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _cardSection({
    required BuildContext context,
    required IconData icon,
    required String title,
    required String subtitle,
    required String buttonText,
    required Color buttonColor,
    required String imageAsset,
    required VoidCallback onPressed,
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
                    onPressed: onPressed,
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

  // Animation for right-to-left transition
  Route _createRoute(Widget page) {
    return PageRouteBuilder(
      pageBuilder: (context, animation, secondaryAnimation) => page,
      transitionsBuilder: (context, animation, secondaryAnimation, child) {
        const begin = Offset(1.0, 0.0);
        const end = Offset.zero;
        const curve = Curves.ease;
        final tween = Tween(begin: begin, end: end).chain(CurveTween(curve: curve));
        return SlideTransition(
          position: animation.drive(tween),
          child: child,
        );
      },
    );
  }
}
