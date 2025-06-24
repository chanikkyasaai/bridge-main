import 'package:canara_ai/screens/nav/home_page.dart';
import 'package:flutter/material.dart';
import 'dart:math';
import 'package:flutter_tts/flutter_tts.dart';

class CaptchaPage extends StatefulWidget {
  const CaptchaPage({Key? key}) : super(key: key);

  @override
  State<CaptchaPage> createState() => _CaptchaPageState();
}

class _CaptchaPageState extends State<CaptchaPage> {
  final TextEditingController _captchaController = TextEditingController();
  List<String> _captchaWords = [];
  final FlutterTts _flutterTts = FlutterTts();

  bool _isVerified = false;
  String _message = '';
  final Random _random = Random();
  final List<String> _wordPool = [
    'apple',
    'banana',
    'cherry',
    'delta',
    'echo',
    'falcon',
    'grape',
    'honey',
    'ice',
    'joker',
    'kiwi',
    'lemon',
    'mango',
    'nectar',
    'orange',
    'peach',
    'quartz',
    'rocket',
    'straw',
    'tango',
    'umbrella',
    'violet',
    'whale',
    'xenon',
    'yellow',
    'zebra',
  ];

  @override
  void initState() {
    super.initState();
    _generateCaptcha();
    _captchaController.addListener(() {
      setState(() {});
    });

    _flutterTts.setLanguage('en-US');
    _flutterTts.setSpeechRate(0.4); // Adjust as needed
    _flutterTts.setPitch(1.0);
  }

  void _generateCaptcha() {
    _captchaWords = List.generate(3, (_) => _wordPool[_random.nextInt(_wordPool.length)]);
    setState(() {
      _message = '';
      _isVerified = false;
      _captchaController.clear();
    });
  }

  void _verifyCaptcha() {
    String userInput = _captchaController.text.trim().toLowerCase();
    String correctAnswer = _captchaWords.join(' ').toLowerCase();

    if (userInput == correctAnswer) {
      setState(() {
        _isVerified = true;
        _message = 'CAPTCHA verified successfully!';
      });
      Navigator.pushAndRemoveUntil(
        context,
        MaterialPageRoute(builder: (context) => const HomePage()),
        (_) => false,
      );
    } else {
      setState(() {
        _isVerified = false;
        _message = 'Invalid CAPTCHA. Please try again.';
        _generateCaptcha();
      });
    }
  }

  Future<void> _speakCaptcha() async {
    if (_captchaWords.isEmpty) return;
    await _flutterTts.stop(); // stop any ongoing speech
    await _flutterTts.speak(_captchaWords.join(', '));
  }

  @override
  void dispose() {
    _captchaController.dispose();
    _flutterTts.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: false,
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        backgroundColor: Colors.grey[100],
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.black),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text(
          'CAPTCHA Verification',
          style: TextStyle(
            color: Colors.black,
            fontSize: 18,
            fontWeight: FontWeight.w500,
          ),
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // CAPTCHA Card
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.1),
                    blurRadius: 10,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  const Text(
                    'Security Verification',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w600,
                      color: Colors.black87,
                    ),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Please enter the words shown below',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      fontSize: 14,
                      color: Colors.grey,
                    ),
                  ),
                  const SizedBox(height: 30),

                  // CAPTCHA Word Display
                  Column(
                    children: [
                      // CAPTCHA word display...
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: _captchaWords
                            .map((word) => Padding(
                                  padding: const EdgeInsets.symmetric(horizontal: 6),
                                  child: Text(
                                    word,
                                    style: TextStyle(
                                      fontSize: 22,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.primaries[_random.nextInt(Colors.primaries.length)],
                                    ),
                                  ),
                                ))
                            .toList(),
                      ),
                      const SizedBox(height: 10),
                      IconButton(
                        icon: const Icon(Icons.volume_up, color: Colors.blue, size: 28),
                        onPressed: _speakCaptcha,
                        tooltip: 'Listen to CAPTCHA',
                      ),
                    ],
                  ),


                  const SizedBox(height: 16),

                  // Refresh Button
                  TextButton.icon(
                    onPressed: _generateCaptcha,
                    icon: const Icon(Icons.refresh, color: Colors.blue, size: 20),
                    label: const Text(
                      'Generate New Words',
                      style: TextStyle(
                        color: Colors.blue,
                        fontSize: 14,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ),

                  const SizedBox(height: 20),

                  // Input Field
                  TextField(
                    controller: _captchaController,
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      fontSize: 18,
                      letterSpacing: 4,
                      fontWeight: FontWeight.w600,
                    ),
                    decoration: InputDecoration(
                      hintText: 'Type the 3 words shown above',
                      hintStyle: TextStyle(
                        color: Colors.grey[400],
                        letterSpacing: 1,
                        fontWeight: FontWeight.normal,
                      ),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8),
                        borderSide: BorderSide(color: Colors.grey[300]!),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8),
                        borderSide: const BorderSide(color: Colors.blue, width: 2),
                      ),
                      contentPadding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 16,
                      ),
                    ),
                    buildCounter: (context, {required currentLength, required isFocused, maxLength}) => null,
                  ),

                  const SizedBox(height: 20),

                  // Verify Button
                  SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      onPressed: _verifyCaptcha,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blue,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                        elevation: 2,
                      ),
                      child: const Text(
                        'Verify',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                  ),

                  const SizedBox(height: 16),

                  // Message Display
                  if (_message.isNotEmpty)
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: _isVerified ? Colors.green[50] : Colors.red[50],
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(
                          color: _isVerified ? Colors.green[200]! : Colors.red[200]!,
                        ),
                      ),
                      child: Row(
                        children: [
                          Icon(
                            _isVerified ? Icons.check_circle : Icons.error,
                            color: _isVerified ? Colors.green : Colors.red,
                            size: 20,
                          ),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              _message,
                              style: TextStyle(
                                color: _isVerified ? Colors.green[700] : Colors.red[700],
                                fontSize: 14,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                ],
              ),
            ),

            const SizedBox(height: 20),

            // Info Card
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.1),
                    blurRadius: 10,
                    offset: const Offset(0, 2),
                  ),
                ],
              ),
              child: Row(
                children: [
                  Icon(
                    Icons.info_outline,
                    color: Colors.blue[600],
                    size: 20,
                  ),
                  const SizedBox(width: 12),
                  const Expanded(
                    child: Text(
                      'Enter the words exactly as shown. The verification is case-insensitive.',
                      style: TextStyle(
                        color: Colors.grey,
                        fontSize: 12,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
