import 'package:canara_ai/utils/input_get_info.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'dart:async';
import 'package:canara_ai/logging/log_touch_data.dart';

class TypingFieldTracker extends StatefulWidget {
  final TextEditingController controller;
  final String fieldName;
  final BehaviorLogger logger;
  final Widget child;
  final String screenName;

  const TypingFieldTracker({
    super.key,
    required this.controller,
    required this.fieldName,
    required this.logger,
    required this.screenName,
    required this.child,
  });

  @override
  State<TypingFieldTracker> createState() => _TypingFieldTrackerState();
}

class _TypingFieldTrackerState extends State<TypingFieldTracker> {
  final List<double> _delays = [];
  int? _prevLength;
  int _deleteCount = 0;

  late DateTime _lastKeyTime;
  late DateTime _typingStartTime;
  DateTime _lastTypingSent = DateTime.now().subtract(const Duration(seconds: 5));
  bool _hasTyped = false;
  int _timeToFirstKey = 0;

  final _channel = const MethodChannel('input_sensor');

  @override
  void initState() {
    super.initState();
    _lastKeyTime = DateTime.now();
    _typingStartTime = DateTime.now();
    widget.controller.addListener(_onChange);
  }

  @override
  void dispose() {
    widget.controller.removeListener(_onChange);
    super.dispose();
  }

  void _onChange() async {
    final now = DateTime.now();

    final delay = now.difference(_lastKeyTime).inMilliseconds / 1000;
    _delays.add(delay);
    _lastKeyTime = now;

    final text = widget.controller.text;
    if (_prevLength != null && text.length < _prevLength!) {
      _deleteCount += (_prevLength! - text.length);
    }
    _prevLength = text.length;

    if (!_hasTyped) {
      _hasTyped = true;
      _timeToFirstKey = now.difference(_typingStartTime).inMilliseconds;
    }

    if (_delays.length >= 5) {
      await _send();
    }
  }

  Future<void> _send() async {
    if (_delays.isEmpty) return;

    final now = DateTime.now();
    if (now.difference(_lastTypingSent) < const Duration(seconds: 3)) return;

    _lastTypingSent = now;
    final speed = _delays.length / _delays.reduce((a, b) => a + b);

    double? pressure, orientation, size;
    try {
      final result = await InputSensor.getTouchInfo();
      pressure = (result?['pressure'] as num?)?.toDouble();
      orientation = (result?['orientation'] as num?)?.toDouble();
      size = (result?['size'] as num?)?.toDouble();
    } catch (_) {}

    final text = widget.controller.text;
    final wordCount = text.trim().isEmpty ? 0 : text.trim().split(RegExp(r'\s+')).length;
    final totalTimeMinutes = now.difference(_typingStartTime).inMilliseconds / 60000.0;
    final wordsPerMinute = totalTimeMinutes > 0 ? wordCount / totalTimeMinutes : 0.0;

    widget.logger.sendEvent('typing_pattern', {
      'screen': widget.screenName,
      'field': widget.fieldName,
      'typing_speed': double.parse(speed.toStringAsFixed(2)),
      'keystroke_dynamics': _delays.map((e) => double.parse(e.toStringAsFixed(3))).toList(),
      'delete_count': _deleteCount,
      'first_key_delay_ms': _timeToFirstKey,
      'total_time_ms': now.difference(_typingStartTime).inMilliseconds,
      'keystroke_count': _delays.length,
      'average_delay': double.parse((_delays.reduce((a, b) => a + b) / _delays.length).toStringAsFixed(2)),
      'words_per_minute': double.parse(wordsPerMinute.toStringAsFixed(2)),
      if (pressure != null) 'touch_pressure': pressure,
      if (orientation != null) 'touch_angle': orientation,
      if (size != null) 'touch_area': size,
    });

    _delays.clear();
  }

  @override
  Widget build(BuildContext context) => widget.child;
}
