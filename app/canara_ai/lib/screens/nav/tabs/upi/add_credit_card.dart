import 'package:canara_ai/screens/nav/tabs/upi/base_upi.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class AddCreditCardPage extends StatefulWidget {
  @override
  _AddCreditCardPageState createState() => _AddCreditCardPageState();
}

class _AddCreditCardPageState extends State<AddCreditCardPage> {
  final _cardNumberController = TextEditingController();
  final _expiryController = TextEditingController();
  final _cvvController = TextEditingController();
  final _nameController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return BasePage(
      title: 'Add RuPay Credit Card',
      child: Container(
        color: Colors.white,
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Card Preview
              Container(
                width: double.infinity,
                height: 200,
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [Colors.purple[600]!, Colors.purple[800]!],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Padding(
                  padding: EdgeInsets.all(20),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text('RuPay', style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
                          Icon(Icons.payments_outlined, color: Colors.white),
                        ],
                      ),
                      Spacer(),
                      Text(
                        _cardNumberController.text.isEmpty ? '**** **** **** ****' : _formatCardNumber(_cardNumberController.text),
                        style: TextStyle(color: Colors.white, fontSize: 20, letterSpacing: 2),
                      ),
                      SizedBox(height: 16),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('CARD HOLDER', style: TextStyle(color: Colors.white70, fontSize: 10)),
                              Text(
                                _nameController.text.isEmpty ? 'YOUR NAME' : _nameController.text.toUpperCase(),
                                style: TextStyle(color: Colors.white, fontSize: 14),
                              ),
                            ],
                          ),
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text('EXPIRES', style: TextStyle(color: Colors.white70, fontSize: 10)),
                              Text(
                                _expiryController.text.isEmpty ? 'MM/YY' : _expiryController.text,
                                style: TextStyle(color: Colors.white, fontSize: 14),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
              SizedBox(height: 32),

              // Form Fields
              TextFormField(
                controller: _cardNumberController,
                keyboardType: TextInputType.number,
                maxLength: 19,
                decoration: InputDecoration(
                  labelText: 'Card Number',
                  hintText: '1234 5678 9012 3456',
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                  prefixIcon: Icon(Icons.credit_card),
                ),
                onChanged: (value) {
                  setState(() {});
                },
                inputFormatters: [
                  FilteringTextInputFormatter.digitsOnly,
                  TextInputFormatter.withFunction((oldValue, newValue) {
                    String text = newValue.text.replaceAll(' ', '');
                    if (text.length <= 16) {
                      text = text.replaceAllMapped(RegExp(r'.{4}'), (match) => '${match.group(0)} ');
                      return TextEditingValue(
                        text: text.trim(),
                        selection: TextSelection.collapsed(offset: text.trim().length),
                      );
                    }
                    return oldValue;
                  }),
                ],
              ),
              SizedBox(height: 16),

              TextFormField(
                controller: _nameController,
                decoration: InputDecoration(
                  labelText: 'Cardholder Name',
                  border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                  prefixIcon: Icon(Icons.person),
                ),
                onChanged: (value) => setState(() {}),
              ),
              SizedBox(height: 16),

              Row(
                children: [
                  Expanded(
                    child: TextFormField(
                      controller: _expiryController,
                      keyboardType: TextInputType.number,
                      maxLength: 5,
                      decoration: InputDecoration(
                        labelText: 'MM/YY',
                        border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                        prefixIcon: Icon(Icons.calendar_today),
                        counterText: '',
                      ),
                      onChanged: (value) {
                        setState(() {});
                        if (value.length == 2 && !value.contains('/')) {
                          _expiryController.text = value + '/';
                          _expiryController.selection = TextSelection.fromPosition(
                            TextPosition(offset: _expiryController.text.length),
                          );
                        }
                      },
                    ),
                  ),
                  SizedBox(width: 16),
                  Expanded(
                    child: TextFormField(
                      controller: _cvvController,
                      keyboardType: TextInputType.number,
                      maxLength: 3,
                      obscureText: true,
                      decoration: InputDecoration(
                        labelText: 'CVV',
                        border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                        prefixIcon: Icon(Icons.security),
                        counterText: '',
                      ),
                    ),
                  ),
                ],
              ),
              SizedBox(height: 24),

              Container(
                padding: EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.blue[50],
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  children: [
                    Icon(Icons.security, color: Colors.blue[600]),
                    SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Your card details are encrypted and secure',
                        style: TextStyle(color: Colors.blue[800]),
                      ),
                    ),
                  ],
                ),
              ),

              Spacer(),

              SizedBox(
                width: double.infinity,
                height: 50,
                child: ElevatedButton(
                  onPressed: () => _addCard(context),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue[600],
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(25)),
                  ),
                  child: Text('Add Card', style: TextStyle(fontSize: 16, color: Colors.white)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  String _formatCardNumber(String number) {
    String digits = number.replaceAll(' ', '');
    if (digits.length <= 4) return digits;
    if (digits.length <= 8) return '${digits.substring(0, 4)} ${digits.substring(4)}';
    if (digits.length <= 12) return '${digits.substring(0, 4)} ${digits.substring(4, 8)} ${digits.substring(8)}';
    return '${digits.substring(0, 4)} ${digits.substring(4, 8)} ${digits.substring(8, 12)} ${digits.substring(12)}';
  }

  void _addCard(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Card Added Successfully'),
        content: Text('Your RuPay credit card has been linked to your UPI account.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: Text('OK'),
          ),
        ],
      ),
    );
  }
}
