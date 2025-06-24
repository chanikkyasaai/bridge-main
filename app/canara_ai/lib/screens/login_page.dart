import 'package:canara_ai/screens/auth_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

import 'register_page.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final TextEditingController _emailController = TextEditingController(text: 'demo@canara.com');
  final TextEditingController _passwordController = TextEditingController(text: 'canara123');

  final Color canaraBlue = const Color(0xFF0072BC);
  final Color canaraYellow = const Color(0xFFFFD600);
  final Color canaraLightBlue = const Color(0xFF00B9F1);
  final Color canaraDarkBlue = const Color(0xFF003366);

  Future<void> _handleLogin()  async {
    final email = _emailController.text.trim();
    final password = _passwordController.text;
    if (email == 'demo@canara.com' && password == 'canara123') {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Login Successful'),
          behavior: SnackBarBehavior.floating,
        ),
      );
      final storage = FlutterSecureStorage();
      await storage.write(key: 'email', value: email);
      await storage.write(key: 'isLoggedIn', value: '1');
      Navigator.pushAndRemoveUntil(
        context,
        MaterialPageRoute(builder: (context) => const AuthPage(isFirst: false,)),
        (route) => false,
      );
      
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Invalid credentials'),
          behavior: SnackBarBehavior.floating,
        ),
      );
    }
  }

  void _navigateToRegister() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const RegisterPage()),
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
              Text('Welcome Back', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: canaraDarkBlue)),
              const SizedBox(height: 8),
              Text('Login using Email and Password', style: TextStyle(fontSize: 16, color: canaraBlue)),
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
              const SizedBox(height: 24),
              SizedBox(
                width: double.infinity,
                child: ElevatedButton(
                  onPressed: _handleLogin,
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    backgroundColor: canaraBlue,
                  ),
                  child: const Text('Login', style: TextStyle(fontSize: 16, color: Colors.white)),
                ),
              ),
              const SizedBox(height: 12),
              TextButton(
                onPressed: _navigateToRegister,
                child: Text('Don\'t have an account? Register', style: TextStyle(color: canaraDarkBlue)),
              ),
            ],
          ),
        ),
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
}
