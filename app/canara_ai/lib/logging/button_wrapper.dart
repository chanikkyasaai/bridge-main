import 'package:canara_ai/logging/log_touch_data.dart';
import 'package:flutter/material.dart';

class LoggedButton extends StatefulWidget {
  final Widget child;
  final String eventName;
  final Map<String, dynamic>? eventData;
  final VoidCallback? onTap;
  final VoidCallback? onLongPress;
  final VoidCallback? onDoubleTap;
  final BehaviorLogger logger;
  final Duration throttleDuration;

  const LoggedButton({
    super.key,
    required this.child,
    required this.eventName,
    required this.logger,
    this.eventData,
    this.onTap,
    this.onLongPress,
    this.onDoubleTap,
    this.throttleDuration = const Duration(seconds: 1),
  });

  @override
  State<LoggedButton> createState() => _LoggedButtonState();
}

class _LoggedButtonState extends State<LoggedButton> {
  DateTime _lastEventTime = DateTime.fromMillisecondsSinceEpoch(0);

  void _log(String type) {
    final now = DateTime.now();
    if (now.difference(_lastEventTime) >= widget.throttleDuration) {
      _lastEventTime = now;
      widget.logger.sendEvent(
        widget.eventName,
        {
          'interaction_type': type,
          ...?widget.eventData,
        },
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        _log('tap');
        widget.onTap?.call();
      },
      onLongPress: () {
        _log('long_press');
        widget.onLongPress?.call();
      },
      onDoubleTap: () {
        _log('double_tap');
        widget.onDoubleTap?.call();
      },
      behavior: HitTestBehavior.translucent,
      child: widget.child,
    );
  }
}
