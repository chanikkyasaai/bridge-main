import 'package:flutter/material.dart';

class SearchSheet extends StatelessWidget {
  final Color canaraBlue;
  final Color canaraDarkBlue;

  const SearchSheet({
    super.key,
    required this.canaraBlue,
    required this.canaraDarkBlue,
  });

  @override
  Widget build(BuildContext context) {
    return DraggableScrollableSheet(
      expand: false,
      initialChildSize: 0.85,
      minChildSize: 0.5,
      maxChildSize: 0.95,
      builder: (context, scrollController) {
        return Column(
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      decoration: InputDecoration(
                        hintText: 'Search',
                        prefixIcon: Icon(Icons.search, color: canaraBlue),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: BorderSide(color: canaraBlue.withOpacity(0.2)),
                        ),
                        contentPadding: const EdgeInsets.symmetric(vertical: 0, horizontal: 12),
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  IconButton(
                    icon: Icon(Icons.close, color: canaraBlue),
                    onPressed: () => Navigator.pop(context),
                  ),
                ],
              ),
            ),
            Expanded(
              child: ListView(
                controller: scrollController,
                children: [
                  _searchListTile('assets/icons/bank.png', 'Send Money'),
                  _searchListTile('assets/icons/money.png', 'Direct Pay'),
                  _searchListTile('assets/icons/mobilephone.png', 'My Beneficiary'),
                  _searchListTile('assets/icons/passbook.png', 'ePassbook'),
                  _searchListTile('assets/icons/b.png', 'Bill Pay'),
                  _searchListTile('assets/icons/send-money.png', 'Card-less Cash'),
                  _searchListTile('assets/icons/bank.png', 'Other Bank Accounts'),
                  _searchListTile('assets/icons/folder.png', 'History'),
                  _searchListTile('assets/icons/mobile-banking.png', 'Manage Accounts'),
                  _searchListTile('assets/icons/contact-book.png', 'Donation'),
                  _searchListTile('assets/icons/user.png', 'My Finance Management'),
                  _searchListTile('assets/icons/register.png', 'Register'),
                ],
              ),
            ),
          ],
        );
      },
    );
  }

  Widget _searchListTile(String asset, String label) {
    return ListTile(
      leading: Image.asset(asset, height: 28, color: canaraBlue),
      title: Text(label, style: TextStyle(color: canaraDarkBlue, fontWeight: FontWeight.w500)),
      onTap: () {},
    );
  }
}
