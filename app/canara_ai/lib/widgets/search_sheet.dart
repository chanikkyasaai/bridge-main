import 'package:canara_ai/logging/log_touch_data.dart';
import 'package:canara_ai/logging/typing_tracker.dart';
import 'package:flutter/material.dart';

class SearchSheet extends StatefulWidget {
  final Color canaraBlue;
  final Color canaraDarkBlue;
  final BehaviorLogger logger;

  const SearchSheet({
    super.key,
    required this.canaraBlue,
    required this.canaraDarkBlue,
    required this.logger
  });

  @override
  State<SearchSheet> createState() => _SearchSheetState();
}

class _SearchSheetState extends State<SearchSheet> {
  final TextEditingController _controller = TextEditingController();
  String _searchQuery = '';

  final List<Map<String, String>> _items = [
    {'icon': 'assets/icons/bank.png', 'label': 'Send Money'},
    {'icon': 'assets/icons/money.png', 'label': 'Direct Pay'},
    {'icon': 'assets/icons/mobilephone.png', 'label': 'My Beneficiary'},
    {'icon': 'assets/icons/passbook.png', 'label': 'ePassbook'},
    {'icon': 'assets/icons/b.png', 'label': 'Bill Pay'},
    {'icon': 'assets/icons/send-money.png', 'label': 'Card-less Cash'},
    {'icon': 'assets/icons/bank.png', 'label': 'Other Bank Accounts'},
    {'icon': 'assets/icons/folder.png', 'label': 'History'},
    {'icon': 'assets/icons/mobile-banking.png', 'label': 'Manage Accounts'},
    {'icon': 'assets/icons/contact-book.png', 'label': 'Donation'},
    {'icon': 'assets/icons/user.png', 'label': 'My Finance Management'},
  ];

  List<Map<String, String>> get _filteredItems {
    if (_searchQuery.isEmpty) return _items;
    return _items.where((item) => item['label']!.toLowerCase().contains(_searchQuery.toLowerCase())).toList();
  }

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Scaffold(
        resizeToAvoidBottomInset: false,
        backgroundColor: Colors.white,
        appBar: AppBar(
          backgroundColor: Colors.white,
          elevation: 0,
          title: const Text('Search', style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold)),
          centerTitle: true,
          iconTheme: IconThemeData(color: widget.canaraBlue),
        ),
        body: SafeArea(
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 18.0, vertical: 12),
                child: TypingFieldTracker(
                  controller: _controller,
                  screenName: 'Search_Sheet',
                  fieldName: 'search_input',
                  logger: widget.logger,
                  child: TextField(
                    controller: _controller,
                    decoration: InputDecoration(
                      hintText: 'Search',
                      hintStyle: TextStyle(color: widget.canaraDarkBlue),
                      prefixIcon: Icon(Icons.search, color: widget.canaraBlue),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                      suffixIcon: _searchQuery.isNotEmpty
                          ? IconButton(
                              icon: Icon(Icons.clear, color: widget.canaraBlue),
                              onPressed: () {
                                _controller.clear();
                                setState(() {
                                  _searchQuery = '';
                                });
                              },
                            )
                          : null,
                    ),
                    onChanged: (value) {
                      setState(() {
                        _searchQuery = value;
                      });
                    },
                  ),
                ),

              ),
              Expanded(
                child: _filteredItems.isEmpty
                    ? Center(
                        child: Text(
                          'No results found',
                          style: TextStyle(color: widget.canaraDarkBlue),
                        ),
                      )
                    : ListView(
                        children: _filteredItems.map((item) => _searchListTile(item['icon']!, item['label']!)).toList(),
                      ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _searchListTile(String asset, String label) {
    return ListTile(
      leading: Image.asset(asset, height: 28, color: widget.canaraBlue),
      title: Text(label, style: TextStyle(color: widget.canaraDarkBlue, fontWeight: FontWeight.w500)),
      onTap: () {},
    );
  }
}
