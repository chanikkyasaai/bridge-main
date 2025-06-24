import 'package:canara_ai/screens/auth_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class RegisterPage extends StatefulWidget {
  const RegisterPage({super.key});

  @override
  State<RegisterPage> createState() => _RegisterPageState();
}

class _RegisterPageState extends State<RegisterPage> {
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _pinController = TextEditingController();

  final Color canaraBlue = const Color(0xFF0072BC);
  final Color canaraYellow = const Color(0xFFFFD600);
  final Color canaraLightBlue = const Color(0xFF00B9F1);
  final Color canaraDarkBlue = const Color(0xFF003366);

  final FocusNode _pinFocusNode = FocusNode();

  Future<void> _handleRegister() async {
    final email = _emailController.text.trim();
    final pass = _passwordController.text;
    final pin = _pinController.text;

    if (email.isEmpty || pass.isEmpty || pin.isEmpty) {
      _showSnack('Please fill all fields');
    } else if (pin.length != 5) {
      _showSnack('PIN must be exactly 5 digits');
    } else {
      _showSnack('Registration Successful');
      final storage = FlutterSecureStorage();
      await storage.write(key: 'email', value: email).then((_) async => await storage.write(key: 'isLoggedIn', value: '1'));
      Navigator.pushAndRemoveUntil(context, MaterialPageRoute(builder: (context) => const AuthPage(isFirst: true,)), (_) => false);
    }
  }

  void _showSnack(String message) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
  }

  Widget _pinWidget() {
    return GestureDetector(
      onTap: () => FocusScope.of(context).requestFocus(_pinFocusNode),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Choose 5-digit PIN'),
          const SizedBox(height: 8),
          Material(
            elevation: 3,
            borderRadius: BorderRadius.circular(12),
            shadowColor: Colors.black.withOpacity(0.2),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(12),
                color: Colors.white,
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: List.generate(5, (index) {
                  String pin = _pinController.text;
                  bool filled = index < pin.length;
                  return Container(
                    width: 28,
                    height: 28,
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
          SizedBox(
            width: 0,
            height: 0,
            child: EditableText(
              controller: _pinController,
              focusNode: _pinFocusNode,
              obscureText: true,
              keyboardType: TextInputType.number,
              style: const TextStyle(color: Colors.transparent, fontSize: 0.1),
              cursorColor: Colors.transparent,
              backgroundCursorColor: Colors.transparent,
              onChanged: (value) => setState(() {}),
            ),
          ),
        ],
      ),
    );
  }

  InputDecoration _inputDecoration(String label, Icon icon, Color color) {
    return InputDecoration(
      labelText: label,
      prefixIcon: icon,
      border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
      focusedBorder: OutlineInputBorder(borderSide: BorderSide(color: color)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: false,
      backgroundColor: Colors.white,
      body: SafeArea(
        minimum: const EdgeInsets.only(top: 5),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Center(child: Image.asset('assets/images/app.webp', height: 80)),
              const SizedBox(height: 5),
              Center(child: Image.asset('assets/images/logo.jpeg', height: 80)),
              const SizedBox(height: 10),
              Text('Create Account', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: canaraDarkBlue)),
              const SizedBox(height: 8),
              Text('Register with your details', style: TextStyle(fontSize: 16, color: canaraBlue)),
              const SizedBox(height: 20),
              TextField(
                controller: _emailController,
                keyboardType: TextInputType.emailAddress,
                decoration: _inputDecoration('Email', const Icon(Icons.email), canaraBlue),
              ),
              const SizedBox(height: 16),
              TextField(
                controller: _passwordController,
                obscureText: true,
                decoration: _inputDecoration('Password', const Icon(Icons.lock), canaraBlue),
              ),
              const SizedBox(height: 20),
              _pinWidget(),
              const SizedBox(height: 24),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _handleRegister,
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    backgroundColor: canaraDarkBlue,
                  ),
                  child: const Text('Register', style: TextStyle(fontSize: 16, color: Colors.white)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
