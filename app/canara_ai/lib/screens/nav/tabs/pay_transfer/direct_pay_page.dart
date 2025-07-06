import 'package:canara_ai/logging/behaviour_route_tracker.dart';
import 'package:canara_ai/logging/log_touch_data.dart';
import 'package:canara_ai/logging/logger_instance.dart';
import 'package:canara_ai/logging/typing_tracker.dart';
import 'package:canara_ai/main.dart';
import 'package:canara_ai/widgets/payment_success.dart';
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';

class DirectPayPage extends StatefulWidget {
  const DirectPayPage({Key? key}) : super(key: key);

  @override
  State<DirectPayPage> createState() => _DirectPayPageState();
}

class _DirectPayPageState extends State<DirectPayPage> with SingleTickerProviderStateMixin {
  late TabController _tabController;
  int _otherBankTab = 1; // 0: Mobile+Bank, 1: Account+IFSC, 2: Mobile+MMID

  // Controllers for fields
  final _accountController = TextEditingController();
  final _reAccountController = TextEditingController();
  final _bankNameController = TextEditingController();
  final _ifscController = TextEditingController();
  final _beneficiaryController = TextEditingController();
  final _nickNameController = TextEditingController();

  late final BehaviorLogger logger;
  late BehaviorRouteTracker tracker;

  final dio = Dio();

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    logger = AppLogger.logger; 
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    tracker = BehaviorRouteTracker(logger, context);
    routeObserver.subscribe(tracker, ModalRoute.of(context)! as PageRoute);
  }

  @override
  void dispose() {
    _tabController.dispose();
    _accountController.dispose();
    _reAccountController.dispose();
    _bankNameController.dispose();
    _ifscController.dispose();
    _beneficiaryController.dispose();
    _nickNameController.dispose();
    super.dispose();

    routeObserver.unsubscribe(tracker);
  }

  Widget _withinCanaraForm() {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        children: [
          TypingFieldTracker(
            controller: _accountController,
            fieldName: 'account_number',
            logger: logger,
            child: TextField(
              controller: _accountController,
              decoration: const InputDecoration(labelText: 'Account number'),
              keyboardType: TextInputType.number,
            ),
          ),
          const SizedBox(height: 12),
          TypingFieldTracker(
            controller: _reAccountController,
            fieldName: 're_account_number',
            logger: logger,
            child: TextField(
              controller: _reAccountController,
              decoration: const InputDecoration(labelText: 'Re-enter Account Number'),
              keyboardType: TextInputType.number,
            ),
          ),
          const SizedBox(height: 12),
          TypingFieldTracker(
            controller: _beneficiaryController,
            fieldName: 'beneficiary_name',
            logger: logger,
            child: TextField(
              controller: _beneficiaryController,
              decoration: const InputDecoration(labelText: 'Beneficiary Name'),
            ),
          ),
          const SizedBox(height: 12),
          TypingFieldTracker(
            controller: _nickNameController,
            fieldName: 'nickname',
            logger: logger,
            child: TextField(
              controller: _nickNameController,
              decoration: const InputDecoration(labelText: 'Nick Name'),
            ),
          ),
          const Spacer(),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              minimumSize: const Size.fromHeight(44),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
              backgroundColor: Colors.blue,
            ),
            onPressed: () {
              Navigator.push(context, MaterialPageRoute(builder: (context) => const PaymentSuccessPage(recipientName: "Rohith", amount: 1000)));
            },
            child: const Text('Confirm', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  Widget _otherBankForm() {
    // Controllers for new fields
    final _mobileController = TextEditingController();
    final _mmidController = TextEditingController();
    final _otherBankBeneficiaryController = TextEditingController();
    final _otherBankNickNameController = TextEditingController();

    Widget mobileBankForm() => Column(
          children: [
            TextField(
              controller: _mobileController,
              decoration: const InputDecoration(
                labelText: 'Mobile Number',
                hintText: 'Enter Beneficiary Mobile Number',
                suffixIcon: Icon(Icons.contacts_outlined),
              ),
              keyboardType: TextInputType.phone,
            ),
            const SizedBox(height: 12),
            DropdownButtonFormField<String>(
              decoration: const InputDecoration(labelText: 'Select Bank', hintText: 'Bank Name'),
              items: ['Canara Bank', 'SBI', 'HDFC', 'ICICI', 'Axis'].map((e) => DropdownMenuItem(value: e, child: Text(e))).toList(),
              onChanged: (v) => _bankNameController.text = v ?? '',
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _otherBankBeneficiaryController,
              decoration: const InputDecoration(
                labelText: 'Beneficiary Name',
                hintText: 'Enter Beneficiary Name',
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _otherBankNickNameController,
              decoration: const InputDecoration(
                labelText: 'Name as per Beneficiary Bank',
              ),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.lightBlue,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
              ),
              onPressed: () {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Dummy: Beneficiary Validated')),
                );
              },
              child: const Text('Validate Beneficiary', style: TextStyle(color: Colors.white)),
            ),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.grey.shade100,
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Text(
                'Disclaimer : Transaction shall be carried out on the basis of Mobile Number and Bank\'s Name entered. The beneficiary name displayed is for reference only. Only 3 beneficiaries can be validated in a day. The transaction can be carried out even in the event of failure of the validation.',
                style: TextStyle(fontSize: 11, color: Colors.black54),
              ),
            ),
            const Spacer(),
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                minimumSize: const Size.fromHeight(44),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
                backgroundColor: Colors.blue,
              ),
              onPressed: () {
                Navigator.push(context, MaterialPageRoute(builder: (context) => const PaymentSuccessPage(recipientName: "Rohith", amount: 1000)));
              },
              child: const Text('Confirm', style: TextStyle(color: Colors.white)),
            ),
          ],
        );

    Widget mobileMmidForm() => Column(
          children: [
            TextField(
              controller: _mobileController,
              decoration: const InputDecoration(
                labelText: 'Mobile Number',
                hintText: 'Enter Beneficiary Mobile Number',
                suffixIcon: Icon(Icons.contacts_outlined),
              ),
              keyboardType: TextInputType.phone,
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _mmidController,
              decoration: const InputDecoration(
                labelText: 'MMID',
                hintText: 'Enter MMID',
              ),
              keyboardType: TextInputType.number,
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _otherBankBeneficiaryController,
              decoration: const InputDecoration(
                labelText: 'Beneficiary Name',
                hintText: 'Name',
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _otherBankNickNameController,
              decoration: const InputDecoration(
                labelText: 'Name as per Beneficiary Bank',
              ),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.lightBlue,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
              ),
              onPressed: () {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Dummy: Beneficiary Validated')),
                );
              },
              child: const Text('Validate Beneficiary', style: TextStyle(color: Colors.white)),
            ),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: Colors.grey.shade100,
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Text(
                'Disclaimer: The Beneficiary name displayed is for reference only. Only 3 beneficiaries can be validated in a day. The transaction can be carried out even in the event of failure of the validation.',
                style: TextStyle(fontSize: 11, color: Colors.black54),
              ),
            ),
            const Spacer(),
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                minimumSize: const Size.fromHeight(44),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
                backgroundColor: Colors.blue,
              ),
              onPressed: () {
                Navigator.push(context, MaterialPageRoute(builder: (context) => const PaymentSuccessPage(recipientName: "Rohith", amount: 1000)));
              },
              child: const Text('Confirm', style: TextStyle(color: Colors.white)),
            ),
          ],
        );

    return Column(
      children: [
        // Segmented control for payment mode
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 8.0),
          child: Row(
            children: [
              _otherBankTabButton('Mobile+Bank', 0),
              _otherBankTabButton('Account + IFSC', 1),
              _otherBankTabButton('Mobile + MMID', 2),
            ],
          ),
        ),
        Expanded(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: _otherBankTab == 0
                ? mobileBankForm()
                : _otherBankTab == 1
                    ? SingleChildScrollView(
                        child: Column(
                          children: [
                            TextField(
                              controller: _accountController,
                              decoration: const InputDecoration(labelText: 'Account Number'),
                              keyboardType: TextInputType.number,
                            ),
                            const SizedBox(height: 12),
                            TextField(
                              controller: _reAccountController,
                              decoration: const InputDecoration(labelText: 'Re-enter Account Number'),
                              keyboardType: TextInputType.number,
                            ),
                            const SizedBox(height: 12),
                            DropdownButtonFormField<String>(
                              decoration: const InputDecoration(labelText: 'Select Bank'),
                              items: ['Canara Bank', 'SBI', 'HDFC', 'ICICI', 'Axis'].map((e) => DropdownMenuItem(value: e, child: Text(e))).toList(),
                              onChanged: (v) => _bankNameController.text = v ?? '',
                            ),
                            const SizedBox(height: 12),
                            TextField(
                              controller: _ifscController,
                              decoration: const InputDecoration(labelText: 'IFSC Code'),
                            ),
                            const SizedBox(height: 12),
                            TextField(
                              controller: _beneficiaryController,
                              decoration: const InputDecoration(labelText: 'Beneficiary Name'),
                            ),
                            const SizedBox(height: 12),
                            TextField(
                              decoration: const InputDecoration(labelText: 'Name as per Beneficiary Bank'),
                            ),
                            const SizedBox(height: 16),
                            ElevatedButton(
                              style: ElevatedButton.styleFrom(
                                backgroundColor: Colors.lightBlue,
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
                              ),
                              onPressed: () {
                                ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(content: Text('Dummy: Beneficiary Validated')),
                                );
                              },
                              child: const Text('Validate Beneficiary', style: TextStyle(color: Colors.white)),
                            ),
                            const SizedBox(height: 8),
                            Container(
                              padding: const EdgeInsets.all(8),
                              decoration: BoxDecoration(
                                color: Colors.grey.shade100,
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: const Text(
                                'Disclaimer: The Beneficiary name displayed is for reference only. Only 3 beneficiaries can be validated in a day. The transaction can be carried out even in the event of failure of the validation.',
                                style: TextStyle(fontSize: 11, color: Colors.black54),
                              ),
                            ),
                            const Spacer(),
                            ElevatedButton(
                              style: ElevatedButton.styleFrom(
                                minimumSize: const Size.fromHeight(44),
                                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
                                backgroundColor: Colors.blue,
                              ),
                              onPressed: () {
                                ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(content: Text('Dummy: Confirmed Other Bank')),
                                );
                              },
                              child: const Text('Confirm', style: TextStyle(color: Colors.white)),
                            ),
                          ],
                        ),
                      )
                    : mobileMmidForm(),
          ),
        ),
      ],
    );
  }

  Widget _otherBankTabButton(String label, int idx) {
    final selected = _otherBankTab == idx;
    return Expanded(
      child: GestureDetector(
        onTap: () => setState(() => _otherBankTab = idx),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: selected ? Colors.blue.shade100 : Colors.transparent,
            border: Border(
              bottom: BorderSide(
                color: selected ? Colors.blue : Colors.grey.shade300,
                width: 2,
              ),
            ),
          ),
          child: Center(
            child: Text(
              label,
              style: TextStyle(
                color: selected ? Colors.blue : Colors.black54,
                fontWeight: selected ? FontWeight.bold : FontWeight.normal,
              ),
            ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: false,
      appBar: AppBar(
        title: const Text('Direct Pay'),
        bottom: TabBar(
          controller: _tabController,
          tabs: const [
            Tab(text: 'Within Canara'),
            Tab(text: 'Other Bank'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _withinCanaraForm(),
          _otherBankForm(),
        ],
      ),
    );
  }
}
