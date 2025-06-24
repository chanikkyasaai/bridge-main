import 'package:canara_ai/screens/captcha_screen.dart';
import 'package:canara_ai/screens/nav/home_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class AuthPage extends StatefulWidget {
  final bool isFirst;
  const AuthPage({super.key, required this.isFirst});

  @override
  State<AuthPage> createState() => _AuthPageState();
}

class _AuthPageState extends State<AuthPage> {
  final TextEditingController _pinController = TextEditingController();
  final FocusNode pinFocusNode = FocusNode();
  bool _showFingerprintDialog = false;

  // Reference image colors
  final Color canaraBlue = const Color(0xFF0072BC);
  final Color canaraYellow = const Color(0xFFFFD600);
  final Color canaraLightBlue = const Color(0xFF00B9F1);
  final Color canaraDarkBlue = const Color(0xFF003366);

  void _showFingerprintPopup(BuildContext context) {
    setState(() {
      _showFingerprintDialog = true;
    });
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Fingerprint'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.fingerprint, size: 64, color: canaraBlue),
            const SizedBox(height: 12),
            const Text('Authenticate with your fingerprint'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () {
              Navigator.of(context).pop();
              setState(() {
                _showFingerprintDialog = false;
              });
            },
            child: const Text('Close'),
          ),
          TextButton(
            onPressed: () async {
              if (widget.isFirst == true) {
                Navigator.pushAndRemoveUntil(context, MaterialPageRoute(builder: (context) => const CaptchaPage()), (route) => false);
              } else {
                Navigator.pushAndRemoveUntil(
                  context,
                  MaterialPageRoute(builder: (context) => const HomePage()),
                  (route) => false,
                );
              }
            },
            child: const Text('Login'),
          ),
        ],
      ),
    );
  }

  Future<void> _dummyLogin(context) async {
    final pin = _pinController.text;
    if (pin == '12345') {
      if (widget.isFirst == true) {
        Navigator.pushAndRemoveUntil(
          context,
          MaterialPageRoute(builder: (context) => const CaptchaPage()),
          (route) => false,
        );
      } else {
        Navigator.pushAndRemoveUntil(
          context,
          MaterialPageRoute(builder: (context) => const HomePage()),
          (route) => false,
        );
      }
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Invalid PIN. Try 123456.')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: false,
      backgroundColor: Colors.white,
      body: SafeArea(
        minimum: EdgeInsets.only(top: 5),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Center(
                child: Image.asset(
                  'assets/images/app.webp',
                  height: 80,
                ),
              ),
              const SizedBox(height: 5),
              Center(
                child: Image.asset(
                  'assets/images/logo.jpeg',
                  height: 80,
                ),
              ),
              const SizedBox(height: 10),
              Text(
                'Hi, Customer',
                style: TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                  color: canaraDarkBlue,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'Login using Passcode / Biometric',
                style: TextStyle(
                  fontSize: 16,
                  color: canaraBlue,
                ),
              ),
              const SizedBox(height: 8),
              _pinwidget(context),
              const SizedBox(height: 5),
              Wrap(
                spacing: 16,
                runSpacing: 16,
                alignment: WrapAlignment.spaceAround,
                children: [
                  _quickAction(Icons.send, 'Send Money', canaraYellow),
                  _quickAction(Icons.qr_code_scanner, 'Scan any UPI QR', canaraLightBlue),
                  _quickAction(Icons.account_balance_wallet, 'View Balance', canaraBlue),
                  _quickAction(Icons.account_box, 'Open A/c', canaraYellow),
                  _quickAction(Icons.account_balance, 'Apply Loan', canaraLightBlue),
                  _quickAction(Icons.local_offer, 'Offers', canaraBlue),
                ],
              ),
              const SizedBox(height: 10),
              // Bottom Banner
              Container(
                margin: const EdgeInsets.symmetric(vertical: 8),
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: canaraLightBlue.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  children: [
                    Icon(Icons.credit_card, color: canaraBlue),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Get your Pre Approved Credit Card Now! Apply now on Canara ai Mobile App!',
                        style: TextStyle(color: canaraBlue),
                      ),
                    ),
                  ],
                ),
              ),
              // Bottom Navigation styled as links in a row
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [_bottomNav(Icons.receipt_long, "Bill Pay", canaraBlue), _bottomNav(Icons.currency_rupee, 'Canara Digital Rupee', canaraYellow), _bottomNav(Icons.help_outline, 'Help', canaraLightBlue), _bottomNav(Icons.more_horiz, "More", canaraDarkBlue)],
                ),
              ),
              const SizedBox(height: 12),
            ],
          ),
        ),
      ),
    );
  }

  Widget _quickAction(IconData icon, String label, Color color) {
    return FittedBox(
        fit: BoxFit.fitHeight,
        child: Container(
          padding: const EdgeInsets.all(5),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(12),
            border: Border.all(
              color: color,
              width: 1,
            ),
            boxShadow: [
              BoxShadow(
                color: Colors.grey.withOpacity(0.2),
                spreadRadius: 2,
                blurRadius: 4,
                offset: const Offset(0, 2),
              ),
            ],
          ),
          child: SizedBox(
              width: 70,
              height: 75,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  CircleAvatar(
                    backgroundColor: color.withOpacity(0.15),
                    child: Icon(icon, color: color),
                  ),
                  const SizedBox(height: 6),
                  Text(label, style: TextStyle(color: color, fontSize: 10), textAlign: TextAlign.center),
                ],
              )),
        ));
  }

  Widget _pinwidget(BuildContext context) {
    return GestureDetector(
      onTap: () => FocusScope.of(context).requestFocus(pinFocusNode),
      child: Column(
        children: [
          // PIN + Fingerprint container
          Container(
            width: 300,
            padding: const EdgeInsets.all(12),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                // PIN Circles with Material elevation
                GestureDetector(
                  onTap: () {
                    FocusScope.of(context).requestFocus(pinFocusNode);
                  },
                  child: Material(
                    elevation: 3,
                    borderRadius: BorderRadius.circular(12),
                    shadowColor: Colors.black.withOpacity(0.2),
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 10),
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(12),
                        color: Colors.white,
                      ),
                      child: GestureDetector(
                        onTap: () => FocusScope.of(context).requestFocus(pinFocusNode),
                        child: Row(
                          children: List.generate(5, (index) {
                            String pin = _pinController.text;
                            bool filled = index < pin.length;
                            return Container(
                              width: 28,
                              height: 28,
                              margin: const EdgeInsets.symmetric(horizontal: 4),
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                border: Border.all(color: canaraBlue, width: 2),
                                color: filled ? canaraBlue : Colors.transparent,
                              ),
                              child: filled
                                  ? Center(
                                      child: Container(
                                        width: 12,
                                        height: 12,
                                        decoration: const BoxDecoration(
                                          color: Colors.white,
                                          shape: BoxShape.circle,
                                        ),
                                      ),
                                    )
                                  : null,
                            );
                          }),
                        ),
                      ),
                    ),
                  ),
                ),

                // Gradient Fingerprint Button
                GestureDetector(
                  onTap: () => _showFingerprintPopup(context),
                  child: Container(
                    width: 48,
                    height: 48,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      gradient: const LinearGradient(
                        colors: [Color(0xFF0D47A1), Color(0xFF1976D2)],
                        begin: Alignment.topLeft,
                        end: Alignment.bottomRight,
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.2),
                          blurRadius: 4,
                          offset: const Offset(0, 2),
                        ),
                      ],
                    ),
                    child: const Icon(Icons.fingerprint, color: Colors.white, size: 28),
                  ),
                ),
              ],
            ),
          ),

          SizedBox(
            width: 0,
            height: 0,
            child: EditableText(
              controller: _pinController,
              focusNode: pinFocusNode,
              obscureText: true,
              keyboardType: TextInputType.number,
              autofocus: true,
              enableInteractiveSelection: false,
              inputFormatters: [
                LengthLimitingTextInputFormatter(5),
                FilteringTextInputFormatter.digitsOnly,
              ],
              style: const TextStyle(color: Colors.transparent, fontSize: 0.1),
              cursorColor: Colors.transparent,
              backgroundCursorColor: Colors.transparent,
              onChanged: (value) {
                setState(() {});
                if (value.length == 5) {
                  _dummyLogin(context);
                }
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _bottomNav(IconData icon, String label, Color color) {
    return SizedBox(
      width: 70, // fixed width for each item
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 50,
            height: 55,
            padding: const EdgeInsets.all(5),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: color,
                width: 1,
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.grey.withOpacity(0.2),
                  spreadRadius: 2,
                  blurRadius: 4,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: CircleAvatar(
              backgroundColor: color.withOpacity(0.15),
              child: Icon(icon, color: color, size: 20),
            ),
          ),
          const SizedBox(height: 6),
          SizedBox(
            height: 30, // fixed height to keep all labels aligned
            child: Text(
              label,
              style: TextStyle(color: color, fontSize: 12),
              textAlign: TextAlign.center,
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }
}
