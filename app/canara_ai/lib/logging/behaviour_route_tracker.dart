import 'package:canara_ai/logging/log_touch_data.dart';
import 'package:flutter/material.dart';

class BehaviorRouteTracker extends RouteAware {
  final BehaviorLogger logger;
  final BuildContext context;

  BehaviorRouteTracker(this.logger, this.context);

  void _log(String transition) {
    final name = ModalRoute.of(context)?.settings.name ?? 'unknown';
    logger.sendEvent('navigation_pattern', {
      'transition': transition,
      'route': name,
    });
  }

  @override
  void didPush() => _log('didPush');
  @override
  void didPop() => _log('didPop');
  @override
  void didPopNext() => _log('didPopNext');
  @override
  void didPushNext() => _log('didPushNext');
}
