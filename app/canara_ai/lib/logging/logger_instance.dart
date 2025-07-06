import 'package:dio/dio.dart';
import 'package:canara_ai/logging/log_touch_data.dart';

class AppLogger {
  static final Dio _dio = Dio();
  static final BehaviorLogger _logger = BehaviorLogger(_dio);

  static BehaviorLogger get logger => _logger;
}
