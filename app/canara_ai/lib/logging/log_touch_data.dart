import 'dart:convert';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:uuid/uuid.dart';
import 'package:flutter/gestures.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

class TouchLogger {
  static final TouchLogger _instance = TouchLogger._internal();
  late final String sessionId;
  late File _logFile;
  final storage = FlutterSecureStorage();

  factory TouchLogger() => _instance;

  TouchLogger._internal() {
    sessionId = const Uuid().v4();
    _initLogFile();
  }

  Future<void> _initLogFile() async {
    final dir = await getApplicationDocumentsDirectory();
    _logFile = File('${dir.path}/touch_log.json');
    if (!await _logFile.exists()) {
      await _logFile.writeAsString(jsonEncode([]));
    }
  }

  Future<void> logEvent(PointerEvent event, String type) async {
    final email = await storage.read(key: 'email');
    final logEntry = {
      'email': email,
      'sessionId': sessionId,
      'timestamp': DateTime.now().toIso8601String(),
      'type': type,
      'position': {
        'x': event.position.dx,
        'y': event.position.dy,
      },
      'pressure': event.pressure,
      'pointer': event.pointer,
      'kind': event.kind.toString(),
    };

    final content = await _logFile.readAsString();
    final List logs = jsonDecode(content);
    logs.add(logEntry);

    print(logs);
    await _logFile.writeAsString(jsonEncode(logs));
  }

  Future<List<Map<String, dynamic>>> readLogs() async {
    final content = await _logFile.readAsString();
    return List<Map<String, dynamic>>.from(jsonDecode(content));
  }

  Future<void> clearLogs() async {
    await _logFile.writeAsString(jsonEncode([]));
  }
}
