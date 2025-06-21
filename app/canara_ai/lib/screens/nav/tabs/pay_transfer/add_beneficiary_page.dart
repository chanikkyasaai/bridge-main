import 'package:flutter/material.dart';

class AddBeneficiaryPage extends StatefulWidget {
  @override
  _AddBeneficiaryPageState createState() => _AddBeneficiaryPageState();
}

class _AddBeneficiaryPageState extends State<AddBeneficiaryPage>
    with SingleTickerProviderStateMixin {
  TabController? _tabController;
  
  // Form Controllers
  final _accountNumberController = TextEditingController();
  final _reEnterAccountController = TextEditingController();
  final _beneficiaryNameController = TextEditingController();
  final _nicknameController = TextEditingController();
  final _mobileController = TextEditingController();
  final _mmidController = TextEditingController();
  final _upiIdController = TextEditingController();
  final _ifscController = TextEditingController();
  
  // Form Keys
  final _formKey = GlobalKey<FormState>();
  
  // State variables
  String _selectedBankType = 'Within Canara';
  bool _isAccountIFSC = false;
  bool _isMobileMMID = true;
  bool _isNameVerified = false;
  
  // Dummy data for banks
  List<Map<String, String>> banks = [
    {'name': 'State Bank of India', 'code': 'SBI'},
    {'name': 'HDFC Bank', 'code': 'HDFC'},
    {'name': 'ICICI Bank', 'code': 'ICICI'},
    {'name': 'Axis Bank', 'code': 'AXIS'},
    {'name': 'Punjab National Bank', 'code': 'PNB'},
    {'name': 'Bank of Baroda', 'code': 'BOB'},
    {'name': 'Union Bank of India', 'code': 'UBI'},
    {'name': 'Kotak Mahindra Bank', 'code': 'KOTAK'},
  ];
  
  String? _selectedBank;
  
  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 4, vsync: this);
  }

  @override
  void dispose() {
    _tabController?.dispose();
    _accountNumberController.dispose();
    _reEnterAccountController.dispose();
    _beneficiaryNameController.dispose();
    _nicknameController.dispose();
    _mobileController.dispose();
    _mmidController.dispose();
    _upiIdController.dispose();
    _ifscController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: false,
      backgroundColor: Color(0xFFF5F5F5),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: Icon(Icons.arrow_back, color: Colors.black),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: Text(
          'Add Beneficiary',
          style: TextStyle(
            color: Colors.black,
            fontSize: 18,
            fontWeight: FontWeight.w500,
          ),
        ),
      ),
      body: Column(
        children: [
          // Tabs
          Container(
            color: Colors.white,
            child: TabBar(
              controller: _tabController,
              labelColor: Colors.black,
              unselectedLabelColor: Colors.grey,
              indicatorColor: Colors.orange,
              isScrollable: true,
              tabs: [
                Tab(text: 'Within Canara'),
                Tab(text: 'Other Bank'),
                Tab(text: 'UPI ID'),
                Tab(text: 'UPI Number'),
              ],
              onTap: (index) {
                setState(() {
                  switch (index) {
                    case 0:
                      _selectedBankType = 'Within Canara';
                      break;
                    case 1:
                      _selectedBankType = 'Other Bank';
                      break;
                    case 2:
                      _selectedBankType = 'UPI ID';
                      break;
                    case 3:
                      _selectedBankType = 'UPI Number';
                      break;
                  }
                });
              },
            ),
          ),
          // Tab Content
          Expanded(
            child: TabBarView(
              controller: _tabController,
              children: [
                _buildWithinCanaraTab(),
                _buildOtherBankTab(),
                _buildUPIIDTab(),
                _buildUPINumberTab(),
              ],
            ),
          ),
          // Bottom Confirm Button
          Container(
            color: Colors.white,
            padding: EdgeInsets.all(16),
            child: SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _handleConfirm,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue,
                  padding: EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(25),
                  ),
                ),
                child: Text(
                  'Confirm',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildWithinCanaraTab() {
    return SingleChildScrollView(
      padding: EdgeInsets.all(16),
      child: Form(
        key: _formKey,
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildTextField(
              controller: _accountNumberController,
              label: 'Account number',
              isRequired: true,
              keyboardType: TextInputType.number,
              validator: (value) {
                if (value == null || value.isEmpty) {
                  return 'Account number is required';
                }
                if (value.length < 10) {
                  return 'Enter valid account number';
                }
                return null;
              },
            ),
            SizedBox(height: 16),
            _buildTextField(
              controller: _reEnterAccountController,
              label: 'Re-enter Account Number',
              isRequired: true,
              keyboardType: TextInputType.number,
              validator: (value) {
                if (value != _accountNumberController.text) {
                  return 'Account numbers do not match';
                }
                return null;
              },
            ),
            SizedBox(height: 16),
            _buildBeneficiaryNameField(),
            SizedBox(height: 16),
            _buildTextField(
              controller: _nicknameController,
              label: 'Nickname',
              isRequired: true,
              validator: (value) {
                if (value == null || value.isEmpty) {
                  return 'Nickname is required';
                }
                return null;
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildOtherBankTab() {
    return SingleChildScrollView(
      padding: EdgeInsets.all(16),
      child: Form(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Toggle buttons
            Row(
              children: [
                Expanded(
                  child: GestureDetector(
                    onTap: () => setState(() => _isAccountIFSC = true),
                    child: Container(
                      padding: EdgeInsets.symmetric(vertical: 12),
                      decoration: BoxDecoration(
                        color: _isAccountIFSC ? Colors.grey.shade300 : Colors.transparent,
                        border: Border.all(color: Colors.grey),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        'Account IFSC',
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: _isAccountIFSC ? Colors.black : Colors.grey,
                        ),
                      ),
                    ),
                  ),
                ),
                SizedBox(width: 12),
                Expanded(
                  child: GestureDetector(
                    onTap: () => setState(() => _isAccountIFSC = false),
                    child: Container(
                      padding: EdgeInsets.symmetric(vertical: 12),
                      decoration: BoxDecoration(
                        color: !_isAccountIFSC ? Colors.blue : Colors.transparent,
                        border: Border.all(color: !_isAccountIFSC ? Colors.blue : Colors.grey),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          if (!_isAccountIFSC) Icon(Icons.check, color: Colors.white, size: 16),
                          SizedBox(width: 4),
                          Text(
                            'Mobile + MMID',
                            style: TextStyle(
                              color: !_isAccountIFSC ? Colors.white : Colors.grey,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            ),
            SizedBox(height: 20),
            
            if (_isAccountIFSC) ...[
              _buildTextField(
                controller: _accountNumberController,
                label: 'Account Number',
                isRequired: true,
                keyboardType: TextInputType.number,
              ),
              SizedBox(height: 16),
              _buildBankDropdown(),
              SizedBox(height: 16),
              _buildTextField(
                controller: _ifscController,
                label: 'IFSC Code',
                isRequired: true,
              ),
            ] else ...[
              _buildBeneficiaryNameField(),
              SizedBox(height: 16),
              _buildTextField(
                controller: _mobileController,
                label: 'Mobile Number',
                isRequired: true,
                keyboardType: TextInputType.phone,
              ),
              SizedBox(height: 16),
              _buildTextField(
                controller: _mmidController,
                label: 'MMID',
                isRequired: true,
                keyboardType: TextInputType.number,
              ),
              SizedBox(height: 16),
              _buildTextField(
                controller: _nicknameController,
                label: 'Nick Name',
                isRequired: true,
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildUPIIDTab() {
    return SingleChildScrollView(
      padding: EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Enter UPI ID',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w500,
              color: Colors.black87,
            ),
          ),
          SizedBox(height: 16),
          _buildTextField(
            controller: _upiIdController,
            label: 'UPI ID',
            isRequired: true,
            hint: 'e.g., username@paytm',
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'UPI ID is required';
              }
              if (!value.contains('@')) {
                return 'Enter valid UPI ID';
              }
              return null;
            },
          ),
          SizedBox(height: 16),
          _buildTextField(
            controller: _nicknameController,
            label: 'Nickname',
            isRequired: true,
          ),
        ],
      ),
    );
  }

  Widget _buildUPINumberTab() {
    return SingleChildScrollView(
      padding: EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Enter Mobile Number',
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.w500,
              color: Colors.black87,
            ),
          ),
          SizedBox(height: 16),
          _buildTextField(
            controller: _mobileController,
            label: 'Mobile Number',
            isRequired: true,
            keyboardType: TextInputType.phone,
            validator: (value) {
              if (value == null || value.isEmpty) {
                return 'Mobile number is required';
              }
              if (value.length != 10) {
                return 'Enter valid 10-digit mobile number';
              }
              return null;
            },
          ),
          SizedBox(height: 16),
          _buildTextField(
            controller: _nicknameController,
            label: 'Nickname',
            isRequired: true,
          ),
        ],
      ),
    );
  }

  Widget _buildTextField({
    required TextEditingController controller,
    required String label,
    bool isRequired = false,
    String? hint,
    TextInputType? keyboardType,
    String? Function(String?)? validator,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        RichText(
          text: TextSpan(
            text: label,
            style: TextStyle(
              color: Colors.black87,
              fontSize: 14,
              fontWeight: FontWeight.w500,
            ),
            children: isRequired
                ? [
                    TextSpan(
                      text: '*',
                      style: TextStyle(color: Colors.red),
                    ),
                  ]
                : [],
          ),
        ),
        SizedBox(height: 8),
        TextFormField(
          controller: controller,
          keyboardType: keyboardType,
          validator: validator,
          decoration: InputDecoration(
            hintText: hint ?? label,
            hintStyle: TextStyle(color: Colors.grey),
            filled: true,
            fillColor: Colors.white,
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(8),
              borderSide: BorderSide(color: Colors.grey.shade300),
            ),
            enabledBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(8),
              borderSide: BorderSide(color: Colors.grey.shade300),
            ),
            focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(8),
              borderSide: BorderSide(color: Colors.blue),
            ),
            contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          ),
        ),
      ],
    );
  }

  Widget _buildBeneficiaryNameField() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        RichText(
          text: TextSpan(
            text: 'Beneficiary Name',
            style: TextStyle(
              color: Colors.black87,
              fontSize: 14,
              fontWeight: FontWeight.w500,
            ),
            children: [
              TextSpan(
                text: '*',
                style: TextStyle(color: Colors.red),
              ),
            ],
          ),
        ),
        SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: TextFormField(
                controller: _beneficiaryNameController,
                decoration: InputDecoration(
                  hintText: 'Name',
                  hintStyle: TextStyle(color: Colors.grey),
                  filled: true,
                  fillColor: Colors.white,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                    borderSide: BorderSide(color: Colors.grey.shade300),
                  ),
                  enabledBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                    borderSide: BorderSide(color: Colors.grey.shade300),
                  ),
                  focusedBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                    borderSide: BorderSide(color: Colors.blue),
                  ),
                  contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                ),
              ),
            ),
            SizedBox(width: 12),
            ElevatedButton(
              onPressed: _verifyBeneficiaryName,
              style: ElevatedButton.styleFrom(
                backgroundColor: _isNameVerified ? Colors.green : Colors.green,
                padding: EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              child: Text(
                _isNameVerified ? 'Verified' : 'Verify',
                style: TextStyle(color: Colors.white),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildBankDropdown() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Select Bank*',
          style: TextStyle(
            color: Colors.black87,
            fontSize: 14,
            fontWeight: FontWeight.w500,
          ),
        ),
        SizedBox(height: 8),
        DropdownButtonFormField<String>(
          value: _selectedBank,
          decoration: InputDecoration(
            filled: true,
            fillColor: Colors.white,
            border: OutlineInputBorder(
              borderRadius: BorderRadius.circular(8),
              borderSide: BorderSide(color: Colors.grey.shade300),
            ),
            enabledBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(8),
              borderSide: BorderSide(color: Colors.grey.shade300),
            ),
            focusedBorder: OutlineInputBorder(
              borderRadius: BorderRadius.circular(8),
              borderSide: BorderSide(color: Colors.blue),
            ),
            contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          ),
          items: banks.map((bank) {
            return DropdownMenuItem<String>(
              value: bank['code'],
              child: Text(bank['name']!),
            );
          }).toList(),
          onChanged: (value) {
            setState(() {
              _selectedBank = value;
            });
          },
          hint: Text('Select Bank'),
        ),
      ],
    );
  }

  void _verifyBeneficiaryName() {
    if (_beneficiaryNameController.text.isNotEmpty) {
      setState(() {
        _isNameVerified = true;
      });
      
      // Simulate verification delay
      Future.delayed(Duration(seconds: 1), () {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Beneficiary name verified successfully!'),
            backgroundColor: Colors.green,
          ),
        );
      });
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Please enter beneficiary name first'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  void _handleConfirm() {
    bool isValid = true;
    String message = '';

    switch (_selectedBankType) {
      case 'Within Canara':
        if (_accountNumberController.text.isEmpty ||
            _reEnterAccountController.text.isEmpty ||
            _beneficiaryNameController.text.isEmpty ||
            _nicknameController.text.isEmpty) {
          isValid = false;
          message = 'Please fill all required fields';
        } else if (_accountNumberController.text != _reEnterAccountController.text) {
          isValid = false;
          message = 'Account numbers do not match';
        }
        break;
        
      case 'Other Bank':
        if (_isAccountIFSC) {
          if (_accountNumberController.text.isEmpty ||
              _selectedBank == null ||
              _ifscController.text.isEmpty) {
            isValid = false;
            message = 'Please fill all required fields';
          }
        } else {
          if (_beneficiaryNameController.text.isEmpty ||
              _mobileController.text.isEmpty ||
              _mmidController.text.isEmpty ||
              _nicknameController.text.isEmpty) {
            isValid = false;
            message = 'Please fill all required fields';
          }
        }
        break;
        
      case 'UPI ID':
        if (_upiIdController.text.isEmpty ||
            _nicknameController.text.isEmpty) {
          isValid = false;
          message = 'Please fill all required fields';
        } else if (!_upiIdController.text.contains('@')) {
          isValid = false;
          message = 'Please enter a valid UPI ID';
        }
        break;
        
      case 'UPI Number':
        if (_mobileController.text.isEmpty ||
            _nicknameController.text.isEmpty) {
          isValid = false;
          message = 'Please fill all required fields';
        } else if (_mobileController.text.length != 10) {
          isValid = false;
          message = 'Please enter a valid 10-digit mobile number';
        }
        break;
    }

    if (isValid) {
      // Show success dialog
      showDialog(
        context: context,
        builder: (BuildContext context) {
          return AlertDialog(
            title: Text('Success'),
            content: Text('Beneficiary added successfully!'),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.of(context).pop(); // Close dialog
                  Navigator.of(context).pop(); // Go back to previous screen
                },
                child: Text('OK'),
              ),
            ],
          );
        },
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(message),
          backgroundColor: Colors.red,
        ),
      );
    }
  }
}