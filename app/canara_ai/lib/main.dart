import 'package:canara_ai/screens/login_page.dart';
import 'package:canara_ai/screens/nav/banking_page.dart';
import 'package:canara_ai/screens/nav/cards/credit_card.dart';
import 'package:canara_ai/screens/nav/cards/debit_card.dart';
import 'package:canara_ai/screens/nav/cards_page.dart';
import 'package:canara_ai/screens/nav/profile/upi_lite_page.dart';
import 'package:canara_ai/screens/nav/profile_page.dart';
import 'package:canara_ai/screens/nav/tabs/banking/manage_accounts_page.dart';
import 'package:canara_ai/screens/nav/tabs/banking/statement_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/add_beneficiary_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/cardless_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/direct_pay_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/donation_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/epassbook_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/history_page.dart';
import 'package:canara_ai/screens/nav/tabs/pay_transfer/send_money_page.dart';
import 'package:canara_ai/screens/nav/tabs/upi/add_credit_card.dart';
import 'package:canara_ai/screens/nav/tabs/upi/approve_payment.dart';
import 'package:canara_ai/screens/nav/tabs/upi/pay_contact.dart';
import 'package:canara_ai/screens/nav/tabs/upi/register_upi.dart';
import 'package:canara_ai/screens/nav/tabs/upi/scan_qr.dart';
import 'package:canara_ai/screens/nav/tabs/upi/tap_pay.dart';
import 'package:canara_ai/widgets/notification_sheet.dart';
import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'screens/auth_page.dart';

final RouteObserver<PageRoute> routeObserver = RouteObserver<PageRoute>();
final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  final FlutterSecureStorage _storage = const FlutterSecureStorage();

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<String?>(
      future: FlutterSecureStorage().read(key: 'isLoggedIn'),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const MaterialApp(
            home: Scaffold(
              body: Center(child: CircularProgressIndicator()),
            ),
          );
        }

        final isLoggedIn = snapshot.data == "true";

        return MaterialApp(
          navigatorKey: navigatorKey,
          debugShowCheckedModeBanner: false,
          title: 'Flutter Demo',
          theme: ThemeData(
            colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
            useMaterial3: true,
          ),
          routes: {
            '/login': (context) => const LoginPage(isFirst: false),
            '/auth': (context) => const AuthPage(isFirst: false),
            '/profile': (context) => const ProfilePage(),
            '/cards': (context) => const CardsPage(),
            '/directpay': (context) => const DirectPayPage(),
            '/notifications': (context) => const NotificationSheet(canaraBlue: Color(0xFF0072BC)),
            '/qrscan': (context) => QRScanPage(),
            '/banking': (context) => const BankingPage(),
            '/addbeneficiary': (context) => AddBeneficiaryPage(),
            '/addcreditcard': (context) => AddCreditCardPage(),
            '/donation': (context) => const DonationPage(),
            '/epassbook': (context) => const EPassbookPage(),
            '/history': (context) => const HistoryPage(),
            '/manage-accounts': (context) => const ManageAccountPage(),
            '/sendmoney': (context) => SendMoneyPage(),
            '/statements': (context) => const StatementPage(),
            '/paycontact': (context) => PayContactPage(),
            '/tappay': (context) => TapPayPage(),
            '/cardlesscash': (context) => const CardlessCashPage(),
            '/registerupi': (context) => RegisterPage(),
            '/upilite': (context) => const UpiLitePage(),
            '/approvepayment': (context) => ApprovePaymentPage(),
            '/creditcards': (context) => CreditCardsPage(),
            '/debitcards': (context) => DebitCardsPage()
          },
          navigatorObservers: [routeObserver],
          home: isLoggedIn ? const AuthPage(isFirst: false) : const LoginPage(isFirst: false),
        );
      },
    );
  }

}