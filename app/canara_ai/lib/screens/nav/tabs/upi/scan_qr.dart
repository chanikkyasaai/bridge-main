import 'package:canara_ai/screens/nav/tabs/upi/base_upi.dart';
import 'package:flutter/material.dart';

class QRScanPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return BasePage(
      title: 'Scan QR Code',
      child: Column(
        children: [
          Expanded(
            child: Container(
              color: Colors.white,
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    width: 250,
                    height: 250,
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.blue[600]!, width: 2),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Stack(
                      children: [
                        Center(
                          child: Icon(
                            Icons.qr_code_scanner,
                            size: 100,
                            color: Colors.blue[600],
                          ),
                        ),
                        // Scanning animation corners
                        Positioned(top: 10, left: 10, child: _scanCorner()),
                        Positioned(top: 10, right: 10, child: _scanCorner(flipX: true)),
                        Positioned(bottom: 10, left: 10, child: _scanCorner(flipY: true)),
                        Positioned(bottom: 10, right: 10, child: _scanCorner(flipX: true, flipY: true)),
                      ],
                    ),
                  ),
                  SizedBox(height: 32),
                  Text(
                    'Position QR code within the frame',
                    style: TextStyle(fontSize: 16, color: Colors.grey[600]),
                  ),
                  SizedBox(height: 16),
                  Text(
                    'Scan any UPI QR code to make instant payments',
                    style: TextStyle(fontSize: 14, color: Colors.grey[500]),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),
          ),
          Container(
            color: Colors.white,
            padding: EdgeInsets.all(16),
            child: Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _simulateQRScan(context),
                    icon: Icon(Icons.photo_library, color: Colors.white),
                    label: Text('Choose from Gallery', style: TextStyle(color: Colors.white)),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue[600],
                      padding: EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(25)),
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

  Widget _scanCorner({bool flipX = false, bool flipY = false}) {
    return Transform.scale(
      scaleX: flipX ? -1 : 1,
      scaleY: flipY ? -1 : 1,
      child: Container(
        width: 20,
        height: 20,
        decoration: BoxDecoration(
          border: Border(
            top: BorderSide(color: Colors.blue[600]!, width: 3),
            left: BorderSide(color: Colors.blue[600]!, width: 3),
          ),
        ),
      ),
    );
  }

  void _simulateQRScan(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('QR Code Detected'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text('Merchant: Coffee Shop'),
            Text('Amount: ₹250.00'),
            Text('UPI ID: coffeeshop@paytm'),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context), child: Text('Cancel')),
          ElevatedButton(onPressed: () => Navigator.pop(context), child: Text('Pay Now')),
        ],
      ),
    );
  }
}
