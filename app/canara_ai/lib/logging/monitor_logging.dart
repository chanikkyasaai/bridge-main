import 'dart:async';
import 'dart:convert';
import 'package:canara_ai/apis/endpoints.dart';
import 'package:canara_ai/apis/interceptor.dart';
import 'package:canara_ai/logging/log_touch_data.dart';
import 'package:canara_ai/logging/sensor_tracker.dart';
import 'package:canara_ai/utils/token_storage.dart';
import 'package:dio/dio.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

/// Widget that monitors user behavior (touch, scroll, idle, typing, app lifecycle)
/// and logs events using the provided [BehaviorLogger].
class BehaviorMonitor extends StatefulWidget {
  final Widget child;
  final BehaviorLogger logger;

  const BehaviorMonitor({super.key, required this.child, required this.logger});

  @override
  State<BehaviorMonitor> createState() => BehaviorMonitorState();
}

class BehaviorMonitorState extends State<BehaviorMonitor> with WidgetsBindingObserver {
  Timer? _idleTimer;
  String? _exitReason;
  final _storage = FlutterSecureStorage();
  final tokenstorage = TokenStorage();
  final dio = Dio();
  String? sessionId;

  /// The sensor tracker instance
  late final SensorTracker sensorTracker;

  //touch params
  DateTime? _lastTouchUpTime;
  DateTime? _touchDownTime;

  //scroll params
  DateTime? _lastScrollTime;
  double? _lastScrollPixels;

  /// Resets the idle timer. If the user is idle for 1 minute, logs an idle event.
  void _resetIdleTimer() {
    _idleTimer?.cancel();
    _idleTimer = Timer(const Duration(minutes: 1), () {
      widget.logger.sendEvent('idle_behavior', {'idle_seconds': 60});
    });
  }

  /// Call this to log a user logout event.
  Future<void> sendUserLogoutEvent() async {
    _exitReason = 'user_logout';
    await _sendExitEvent();
    // await widget.logger.endSession(_exitReason!);
  }

  /// Sends an exit event (logout, app close, etc.) to the backend.
  /// If sending fails, saves the event locally for retry.
  Future<void> _sendExitEvent() async {
    final payload = {'session_id': sessionId ?? widget.logger.sessionId, 'reason': _exitReason ?? 'app_close', 'session_token': widget.logger.sessionToken};

    try {
      // Ensure Dio has up-to-date token access
      dio.interceptors.clear(); // avoid duplicate interceptors
      dio.interceptors.add(AuthInterceptor(dio, tokenstorage));

      final response = await dio.post('${Endpoints.baseUrl}${Endpoints.log_exit}', data: payload);

      if (response.statusCode == 200) {
        await _storage.delete(key: 'pending_exit_event');
        debugPrint('Exit event sent successfully');
      } else {
        await _storage.write(key: 'pending_exit_event', value: jsonEncode(payload));
        debugPrint('Exit event failed. Status: ${response.statusCode}. Saved for retry.');
      }
    } catch (e) {
      debugPrint('Exit event error: $e. Saving to retry.');
      await _storage.write(key: 'pending_exit_event', value: jsonEncode(payload));
    }
  }

  Future<void> checkForPreviousForceClose() async {
    String? lastCloseState = await _storage.read(key: 'app_closed_properly');
    if (lastCloseState != 'true') {
      _exitReason = 'force_close';
      _sendExitEvent(); // Optional — inform backend retroactively
    }

    await _storage.write(key: 'app_closed_properly', value: 'false'); // Reset flag
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);

    sessionId = widget.logger.sessionId;
    _resetIdleTimer();

    sensorTracker = SensorTracker(widget.logger);
    sensorTracker.start();
  }

  @override
  void dispose() {
    _idleTimer?.cancel();
    WidgetsBinding.instance.removeObserver(this);

    sensorTracker.stop();

    super.dispose();
  }

  /// Handles app lifecycle changes and logs navigation/exit events.
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    super.didChangeAppLifecycleState(state);

    String? reason;

    switch (state) {
      case AppLifecycleState.inactive:
        // Not used for exit, maybe in-call, not quitting
        break;

      case AppLifecycleState.paused:
        // App goes to background
        reason = 'app_background';
        break;

      case AppLifecycleState.resumed:
        // App comes to foreground
        reason = 'app_foreground';
        break;

      case AppLifecycleState.detached:
        // App closed or terminated (e.g. swipe to close)
        reason = 'app_close';
        break;
      case AppLifecycleState.hidden:
        // App goes to background
        reason = 'app_background';
        break;
    }

    if (reason != null) {
      _exitReason = reason;
      _sendExitEvent();
    }

    widget.logger.sendEvent('navigation_pattern', {'state': state.toString()});
  }

  @override
  Widget build(BuildContext context) {
    // Listens for all pointer (touch) events and logs them.
    return Listener(
      behavior: HitTestBehavior.translucent,
      onPointerDown: (PointerDownEvent event) {
        _resetIdleTimer();

        _touchDownTime = DateTime.now();

        final now = _touchDownTime!;
        final interTouch = (_lastTouchUpTime != null) ? now.difference(_lastTouchUpTime!).inMilliseconds : null;

        final position = event.position;
        final pressure = event.pressure;

        widget.logger.sendEvent('touch_down', {
          'coordinates': [position.dx, position.dy],
          'pressure': pressure,
          if (interTouch != null) 'inter_touch_gap_ms': interTouch,
        });
      },
      onPointerUp: (PointerUpEvent event) {
        final now = DateTime.now();
        final duration = (_touchDownTime != null) ? now.difference(_touchDownTime!).inMilliseconds : null;

        _lastTouchUpTime = now;

        final position = event.position;

        widget.logger.sendEvent('touch_up', {
          'coordinates': [position.dx, position.dy],
          if (duration != null) 'touch_duration_ms': duration,
        });
      },
      // Listens for scroll notifications and logs them (throttled).
      child: NotificationListener<ScrollNotification>(
        onNotification: (scroll) {
          final now = DateTime.now();
          final current = scroll.metrics.pixels;

          double? velocity;
          if (_lastScrollTime != null && _lastScrollPixels != null) {
            final dt = now.difference(_lastScrollTime!).inMilliseconds / 1000.0;
            velocity = (current - _lastScrollPixels!) / dt;
          }

          _lastScrollTime = now;
          _lastScrollPixels = current;

          widget.logger.sendEvent('scroll', {
            'pixels': current,
            'max': scroll.metrics.maxScrollExtent,
            if (velocity != null) 'velocity': velocity,
          });

          return false;
        },

        child: Builder(
          builder: (context) {
            sensorTracker.trackOrientation(context);
            sensorTracker.trackBrightness(context);
            return widget.child;
          },
        ),
      ),
    );
  }
}
