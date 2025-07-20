import 'package:flutter/services.dart';

class InputSensor {
  static const MethodChannel _channel = MethodChannel('input_sensor');

  static Future<Map<String, dynamic>?> getTouchInfo() async {
    try {
      final result = await _channel.invokeMethod('getTouchInfo');
      return Map<String, dynamic>.from(result);
    } catch (e) {
      print('Failed to get touch info: $e');
      return null;
    }
  }
}