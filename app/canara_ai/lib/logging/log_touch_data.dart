import 'dart:async';
import 'dart:convert';
import 'package:canara_ai/apis/endpoints.dart';
import 'package:canara_ai/apis/interceptor.dart';
import 'package:canara_ai/utils/token_storage.dart';
import 'package:dio/dio.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/status.dart' as status;

class BehaviorLogger {
  final Dio dio;
  WebSocketChannel? _ws;
  String? sessionId;
  String? sessionToken;

  final tokenstorage = TokenStorage();

  BehaviorLogger(this.dio);

  void _startHeartbeat() {
    Timer.periodic(Duration(seconds: 30), (timer) {
      if (_ws == null) {
        timer.cancel();
        return;
      }
      _ws!.sink.add(jsonEncode({'type': 'ping', 'timestamp': DateTime.now().toIso8601String()}));
    });
  }

  final _eventQueue = StreamController<String>();
  bool _isSending = false;

  void _startEventQueue() {
    if (_isSending) return;

    _isSending = true;
    _eventQueue.stream.listen((eventJson) async {
      try {
        _ws?.sink.add(eventJson);
      } catch (e) {
        print("WebSocket send error: $e");
      }
    });
  }


  Future<void> startSession(String _sessionId, String _sessionToken) async {
    // dio.interceptors.add(AuthInterceptor(dio, tokenstorage));
    // final deviceInfo = await getExtendedAndroidDeviceSummary();
    // final deviceid = await getAndroidDeviceId();
    // final sessionId = const Uuid().v4();

    // final response = await dio.post('${Endpoints.baseUrl}${Endpoints.log_start}', data: {
    //   'phone': phone,
    //   'device_id': deviceid,
    //   'device_info': deviceInfo,
    // });

    final token = (await tokenstorage.getAccessToken())?.replaceAll('#', '');
    dio.options.headers['Authorization'] = 'Bearer $token';

    print('Session ID: $_sessionId');
    print('Session Token: $_sessionToken');

    sessionId = _sessionId;
    sessionToken = _sessionToken;

    final uri = Uri.parse('ws://35.225.176.106:8000/api/v1/ws/behavior/$_sessionId?token=${Uri.encodeComponent(_sessionToken)}');

    print(uri);

    _ws = WebSocketChannel.connect(uri);
    _startEventQueue();

    _ws!.stream.listen(
      _handleServerMessage,
      onDone: () {
        print('Connection closed by server');
        _retryConnection();
      },
      onError: (err) {
        print('Connection error: $err');
        _retryConnection();
      },
    );

    _startHeartbeat();
    print("WebSocket connected and heartbeat started");
  }

  void sendEvent(String eventType, Map<String, dynamic> data) {
    print('Sending event: $eventType');
    print(data);
    if (_ws == null || sessionId == null) return;

    final payload = {
      'event_type': eventType,
      'data': data,
      'timestamp': DateTime.now().toIso8601String(),
    };

    if (eventType == 'navigation_pattern' && data['route'] == 'unknown') {
      return;
    }

    print("Payload loaded");
    try {
      print("Payload sent : $payload");
      final eventJson = jsonEncode(payload);
      print("Queueing event: $eventType");
      _eventQueue.add(eventJson);
    } catch (e) {
      print(payload);
      print(e);
    }
  }

  Future<void> endSession(String finalDecision) async {
    dio.interceptors.add(AuthInterceptor(dio, tokenstorage));
    if (sessionId != null) {
      await dio.post('${Endpoints.baseUrl}${Endpoints.log_end}', data: {
        'session_id': sessionId,
        'final_decision': finalDecision,
        'session_token': sessionToken
      });
    }

    await _ws?.sink.close(status.normalClosure);
    _ws = null;
    sessionId = null;
  }

  void _handleServerMessage(dynamic message) {
    final data = jsonDecode(message);
    if (data['type'] == 'connection_established') {
      print('WebSocket connected: ${data['message']}');
    } else if (data['type'] == 'data_received') {
      print('Event processed: ${data['timestamp']}');
    } else if (data['type'] == 'error') {
      print('WebSocket error: ${data['message']}');
    }
  }

  void _retryConnection({int retries = 5}) async {
    for (int attempt = 0; attempt < retries; attempt++) {
      await Future.delayed(const Duration(seconds: 3));
      try {
        if (sessionId == null) return;

        print('Session ID: $sessionId');
        print('Session Token: $sessionToken');
        final token = await tokenstorage.getAccessToken();
        final uri = Uri.parse('ws://35.225.176.106:8000/api/v1/ws/behavior/$sessionId?token=${Uri.encodeComponent(sessionToken!)}');

        _ws = WebSocketChannel.connect(uri);

        _ws!.stream.listen(
          _handleServerMessage,
          onDone: () => _retryConnection(retries: retries - 1),
          onError: (err) => _retryConnection(retries: retries - 1),
          cancelOnError: true,
        );
        print("WebSocket reconnected.");
        return;
      } catch (_) {
        print("Reconnect attempt $attempt failed");
      }
    }
    print("Max reconnect attempts reached");
  }
}
