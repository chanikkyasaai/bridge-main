import 'package:flutter/material.dart';

class SendMoneyPage extends StatefulWidget {
  @override
  _SendMoneyPageState createState() => _SendMoneyPageState();
}

class _SendMoneyPageState extends State<SendMoneyPage>
    with SingleTickerProviderStateMixin {
  TabController? _tabController;
  TextEditingController _searchController = TextEditingController();
  String selectedTransferType = 'Within Canara';

  // Dummy data
  List<Map<String, dynamic>> beneficiaries = [
    {
      'name': 'John Doe',
      'upiId': 'john.doe@paytm',
      'mobile': '9876543210',
      'bankName': 'Canara Bank',
      'accountNumber': '****1234',
      'isFavorite': true,
    },
    {
      'name': 'Alice Smith',
      'upiId': 'alice.smith@gpay',
      'mobile': '9876543211',
      'bankName': 'HDFC Bank',
      'accountNumber': '****5678',
      'isFavorite': false,
    },
    {
      'name': 'Bob Johnson',
      'upiId': 'bob.johnson@phonepe',
      'mobile': '9876543212',
      'bankName': 'Canara Bank',
      'accountNumber': '****9012',
      'isFavorite': true,
    },
    {
      'name': 'Sarah Wilson',
      'upiId': 'sarah.wilson@ybl',
      'mobile': '9876543213',
      'bankName': 'SBI',
      'accountNumber': '****3456',
      'isFavorite': false,
    },
  ];

  List<Map<String, dynamic>> recentTransactions = [
    {
      'name': 'Mike Davis',
      'upiId': 'mike.davis@paytm',
      'amount': '₹500',
      'date': '2 hours ago',
      'type': 'Sent',
    },
    {
      'name': 'Emma Brown',
      'upiId': 'emma.brown@gpay',
      'amount': '₹1,200',
      'date': 'Yesterday',
      'type': 'Received',
    },
  ];

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
  }

  @override
  void dispose() {
    _tabController?.dispose();
    _searchController.dispose();
    super.dispose();
  }

  List<Map<String, dynamic>> getFilteredBeneficiaries() {
    String query = _searchController.text.toLowerCase();
    List<Map<String, dynamic>> filtered = beneficiaries.where((beneficiary) {
      bool matchesSearch = beneficiary['name'].toLowerCase().contains(query) ||
          beneficiary['upiId'].toLowerCase().contains(query) ||
          beneficiary['mobile'].contains(query);

      if (selectedTransferType == 'Within Canara') {
        return matchesSearch && beneficiary['bankName'] == 'Canara Bank';
      } else if (selectedTransferType == 'Other Bank') {
        return matchesSearch && beneficiary['bankName'] != 'Canara Bank';
      }
      return matchesSearch;
    }).toList();

    return filtered;
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
          'Send Money',
          style: TextStyle(
            color: Colors.black,
            fontSize: 18,
            fontWeight: FontWeight.w500,
          ),
        ),
      ),
      body: Column(
        children: [
          // Header with UPI ID
          Container(
            color: Colors.white,
            padding: EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'My UPI ID: chanikkyasaai@cnrb',
                  style: TextStyle(
                    color: Colors.blue,
                    fontSize: 14,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                SizedBox(height: 16),
                // Search Bar
                Container(
                  decoration: BoxDecoration(
                    color: Color(0xFFF5F5F5),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: TextField(
                    controller: _searchController,
                    onChanged: (value) => setState(() {}),
                    decoration: InputDecoration(
                      hintText: 'Search Beneficiary',
                      hintStyle: TextStyle(color: Colors.grey),
                      prefixIcon: Icon(Icons.search, color: Colors.grey),
                      border: InputBorder.none,
                      contentPadding: EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 12,
                      ),
                    ),
                  ),
                ),
                SizedBox(height: 8),
                // Sync IB Beneficiary
                TextButton(
                  onPressed: () {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text('Syncing beneficiaries...')),
                    );
                  },
                  child: Text(
                    'Sync IB Beneficiary',
                    style: TextStyle(color: Colors.blue),
                  ),
                ),
              ],
            ),
          ),
          // Tabs
          Container(
            color: Colors.white,
            child: TabBar(
              controller: _tabController,
              labelColor: Colors.black,
              unselectedLabelColor: Colors.grey,
              indicatorColor: Colors.orange,
              tabs: [
                Tab(text: 'Mobile Banking'),
                Tab(text: 'UPI'),
                Tab(text: 'Favourites'),
              ],
            ),
          ),
          // Tab Content
          Expanded(
            child: TabBarView(
              controller: _tabController,
              children: [
                _buildMobileBankingTab(),
                _buildUPITab(),
                _buildFavouritesTab(),
              ],
            ),
          ),
          // Bottom Buttons
          Container(
            color: Colors.white,
            padding: EdgeInsets.all(16),
            child: Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: () {
                      _showAddBeneficiaryDialog();
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue,
                      padding: EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(25),
                      ),
                    ),
                    child: Text(
                      'Add Beneficiary',
                      style: TextStyle(color: Colors.white),
                    ),
                  ),
                ),
                SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton(
                    onPressed: () {
                      _showDirectPayDialog();
                    },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue,
                      padding: EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(25),
                      ),
                    ),
                    child: Text(
                      'Direct Pay',
                      style: TextStyle(color: Colors.white),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMobileBankingTab() {
    return Column(
      children: [
        // Transfer Type Selection
        Container(
          color: Colors.white,
          padding: EdgeInsets.all(16),
          child: Row(
            children: [
              _buildTransferTypeButton('Within Canara'),
              SizedBox(width: 8),
              _buildTransferTypeButton('Other Bank'),
              SizedBox(width: 8),
              _buildTransferTypeButton('Own Account'),
            ],
          ),
        ),
        // Beneficiary List
        Expanded(
          child: _buildBeneficiaryList(),
        ),
      ],
    );
  }

  Widget _buildUPITab() {
    return Column(
      children: [
        // UPI Options
        Container(
          color: Colors.white,
          padding: EdgeInsets.all(16),
          child: Row(
            children: [
              Expanded(
                child: ElevatedButton(
                  onPressed: () => setState(() {}),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.blue,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(25),
                    ),
                  ),
                  child: Text(
                    'Contact/Mobile Number',
                    style: TextStyle(color: Colors.white, fontSize: 12),
                  ),
                ),
              ),
              SizedBox(width: 8),
              Expanded(
                child: OutlinedButton(
                  onPressed: () => setState(() {}),
                  style: OutlinedButton.styleFrom(
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(25),
                    ),
                  ),
                  child: Text(
                    'UPI ID',
                    style: TextStyle(color: Colors.grey),
                  ),
                ),
              ),
            ],
          ),
        ),
        // Pay to Contact Section
        Container(
          color: Colors.white,
          padding: EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Pay to Contact/Mobile Number',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w500,
                ),
              ),
              SizedBox(height: 12),
              Container(
                decoration: BoxDecoration(
                  color: Color(0xFFF5F5F5),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: TextField(
                  decoration: InputDecoration(
                    hintText: 'Search or Enter UPI/Mobile Number',
                    hintStyle: TextStyle(color: Colors.grey),
                    suffixIcon: Icon(Icons.contacts, color: Colors.blue),
                    border: InputBorder.none,
                    contentPadding: EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 12,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
        // Beneficiary List
        Expanded(
          child: _buildBeneficiaryList(),
        ),
      ],
    );
  }

  Widget _buildFavouritesTab() {
    List<Map<String, dynamic>> favourites = beneficiaries
        .where((beneficiary) => beneficiary['isFavorite'])
        .toList();

    return _buildBeneficiaryList(favourites);
  }

  Widget _buildTransferTypeButton(String type) {
    bool isSelected = selectedTransferType == type;
    return Expanded(
      child: GestureDetector(
        onTap: () {
          setState(() {
            selectedTransferType = type;
          });
        },
        child: Container(
          padding: EdgeInsets.symmetric(vertical: 8, horizontal: 12),
          decoration: BoxDecoration(
            color: isSelected ? Colors.blue : Colors.transparent,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(
              color: isSelected ? Colors.blue : Colors.grey,
            ),
          ),
          child: Text(
            type,
            textAlign: TextAlign.center,
            style: TextStyle(
              color: isSelected ? Colors.white : Colors.grey,
              fontSize: 12,
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildBeneficiaryList([List<Map<String, dynamic>>? customList]) {
    List<Map<String, dynamic>> listToShow = customList ?? getFilteredBeneficiaries();
    
    if (listToShow.isEmpty) {
      return Container(
        color: Colors.white,
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.people_outline,
                size: 64,
                color: Colors.grey,
              ),
              SizedBox(height: 16),
              Text(
                'There are no Beneficiary to show right now',
                style: TextStyle(
                  color: Colors.grey,
                  fontSize: 16,
                ),
              ),
            ],
          ),
        ),
      );
    }

    return Container(
      color: Colors.white,
      child: ListView.builder(
        itemCount: listToShow.length,
        itemBuilder: (context, index) {
          final beneficiary = listToShow[index];
          return ListTile(
            leading: CircleAvatar(
              backgroundColor: Colors.blue,
              child: Text(
                beneficiary['name'][0].toUpperCase(),
                style: TextStyle(color: Colors.white),
              ),
            ),
            title: Text(
              beneficiary['name'],
              style: TextStyle(fontWeight: FontWeight.w500),
            ),
            subtitle: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(beneficiary['upiId']),
                Text('${beneficiary['bankName']} • ${beneficiary['accountNumber']}'),
              ],
            ),
            trailing: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (beneficiary['isFavorite'])
                  Icon(Icons.star, color: Colors.orange, size: 20),
                Icon(Icons.arrow_forward_ios, size: 16, color: Colors.grey),
              ],
            ),
            onTap: () {
              _showPaymentDialog(beneficiary);
            },
          );
        },
      ),
    );
  }

  void _showAddBeneficiaryDialog() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Add Beneficiary'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                decoration: InputDecoration(
                  labelText: 'Name',
                  border: OutlineInputBorder(),
                ),
              ),
              SizedBox(height: 12),
              TextField(
                decoration: InputDecoration(
                  labelText: 'UPI ID',
                  border: OutlineInputBorder(),
                ),
              ),
              SizedBox(height: 12),
              TextField(
                decoration: InputDecoration(
                  labelText: 'Mobile Number',
                  border: OutlineInputBorder(),
                ),
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.of(context).pop();
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Beneficiary added successfully!')),
                );
              },
              child: Text('Add'),
            ),
          ],
        );
      },
    );
  }

  void _showDirectPayDialog() {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Direct Pay'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                decoration: InputDecoration(
                  labelText: 'UPI ID or Mobile Number',
                  border: OutlineInputBorder(),
                ),
              ),
              SizedBox(height: 12),
              TextField(
                decoration: InputDecoration(
                  labelText: 'Amount',
                  border: OutlineInputBorder(),
                  prefixText: '₹ ',
                ),
                keyboardType: TextInputType.number,
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.of(context).pop();
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Payment initiated!')),
                );
              },
              child: Text('Pay'),
            ),
          ],
        );
      },
    );
  }

  void _showPaymentDialog(Map<String, dynamic> beneficiary) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Send Money to ${beneficiary['name']}'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text('UPI ID: ${beneficiary['upiId']}'),
              SizedBox(height: 12),
              TextField(
                decoration: InputDecoration(
                  labelText: 'Amount',
                  border: OutlineInputBorder(),
                  prefixText: '₹ ',
                ),
                keyboardType: TextInputType.number,
              ),
              SizedBox(height: 12),
              TextField(
                decoration: InputDecoration(
                  labelText: 'Remark (Optional)',
                  border: OutlineInputBorder(),
                ),
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.of(context).pop(),
              child: Text('Cancel'),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.of(context).pop();
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text('Payment sent to ${beneficiary['name']}!')),
                );
              },
              child: Text('Send'),
            ),
          ],
        );
      },
    );
  }
}