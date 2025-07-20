import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:flutter/services.dart';
import 'package:canara_ai/logging/log_touch_data.dart';
import 'package:screen_brightness/screen_brightness.dart';

class SensorTracker with WidgetsBindingObserver {
  final BehaviorLogger logger;

  late final StreamSubscription<GyroscopeEvent> _gyroSub;
  late final StreamSubscription<AccelerometerEvent> _accelSub;

  DateTime _lastGyroSent = DateTime.now().subtract(const Duration(seconds: 3));
  DateTime _lastAccelSent = DateTime.now().subtract(const Duration(seconds: 3));
  DateTime _lastOrientationLog = DateTime.now().subtract(const Duration(seconds: 5));
  DateTime _lastBrightnessLog = DateTime.now().subtract(const Duration(seconds: 5));

  Orientation? _lastOrientation;

  SensorTracker(this.logger);

  void start() {
    WidgetsBinding.instance.addObserver(this);

    _gyroSub = gyroscopeEvents.listen((event) {
      final now = DateTime.now();
      final rotationMagnitude = sqrt(event.x * event.x + event.y * event.y + event.z * event.z);

      if (rotationMagnitude > 10) {
        logger.sendEvent('fast_rotation', {
          'x': event.x,
          'y': event.y,
          'z': event.z,
          'magnitude': rotationMagnitude,
        });
      }

      if (now.difference(_lastGyroSent) > const Duration(seconds: 3)) {
        _lastGyroSent = now;
        logger.sendEvent('gyro_data', {
          'x': event.x,
          'y': event.y,
          'z': event.z,
        });
      }
    });

    _accelSub = accelerometerEvents.listen((event) {
      final now = DateTime.now();
      final accMagnitude = sqrt(event.x * event.x + event.y * event.y + event.z * event.z);

      if (accMagnitude > 20) {
        logger.sendEvent('shake_detected', {
          'x': event.x,
          'y': event.y,
          'z': event.z,
          'magnitude': accMagnitude,
        });
      }

      if (event.z < -9.5) {
        logger.sendEvent('flip_detected', {
          'z': event.z,
        });
      }

      if (now.difference(_lastAccelSent) > const Duration(seconds: 3)) {
        _lastAccelSent = now;
        logger.sendEvent('accel_data', {
          'x': event.x,
          'y': event.y,
          'z': event.z,
        });
      }
    });
  }

  void trackOrientation(BuildContext context) {
    final now = DateTime.now();
    final orientation = MediaQuery.of(context).orientation;
    if (orientation != _lastOrientation && now.difference(_lastOrientationLog) > const Duration(seconds: 3)) {
      _lastOrientation = orientation;
      _lastOrientationLog = now;

      logger.sendEvent('orientation_change', {
        'orientation': orientation == Orientation.portrait ? 'portrait' : 'landscape',
      });
    }
  }

  void trackBrightness(BuildContext context) async {
    final now = DateTime.now();
    if (now.difference(_lastBrightnessLog) < const Duration(seconds: 3)) return;

    _lastBrightnessLog = now;

    try {
      final brightnessLevel = await ScreenBrightness().current;

      logger.sendEvent('brightness_change', {
        'theme': MediaQuery.of(context).platformBrightness == Brightness.dark ? 'dark' : 'light',
        'brightness_level': double.parse(brightnessLevel.toStringAsFixed(2)), // 0.0 to 1.0
      });
    } catch (e) {
      print("Failed to get brightness: $e");
    }
  }

  void stop() {
    _gyroSub.cancel();
    _accelSub.cancel();
    WidgetsBinding.instance.removeObserver(this);
  }
}
