import 'dart:io';
import 'package:device_info_plus/device_info_plus.dart';

Future<String?> getAndroidDeviceId() async {
  if (Platform.isAndroid) {
    final deviceInfo = DeviceInfoPlugin();
    final androidInfo = await deviceInfo.androidInfo;
    return androidInfo.id; // This is the Android device ID (SSAID)
  }
  return null;
}

Future<String> getExtendedAndroidDeviceSummary() async {
  final deviceInfo = DeviceInfoPlugin();

  if (Platform.isAndroid) {
    final info = await deviceInfo.androidInfo;

    final osVersion = info.version.release;
    final sdkInt = info.version.sdkInt;
    final manufacturer = info.manufacturer;
    final model = info.model;
    final brand = info.brand;
    final product = info.product;
    final hardware = info.hardware;

    return '''
      Android $osVersion (SDK $sdkInt),
      Device: $manufacturer $model,
      Brand: $brand, Hardware: $hardware,
      '''
        .trim();
  }

  return 'Unsupported platform';
}
