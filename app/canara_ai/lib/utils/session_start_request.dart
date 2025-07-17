import 'package:device_info_plus/device_info_plus.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:geolocator/geolocator.dart';
import 'package:uuid/uuid.dart';
import 'dart:io';
import 'package:flutter/foundation.dart';

class SessionStartRequest {
  final String sessionId;
  final String userId;
  final String deviceId;
  final String phone;
  final String deviceType;
  final String deviceModel;
  final String osVersion;
  final String appVersion;
  final String networkType;
  final Map<String, dynamic> locationData;
  final bool isKnownDevice;
  final bool isTrustedLocation;

  SessionStartRequest({
    required this.sessionId,
    required this.userId,
    required this.deviceId,
    required this.phone,
    required this.deviceType,
    required this.deviceModel,
    required this.osVersion,
    required this.appVersion,
    required this.networkType,
    required this.locationData,
    required this.isKnownDevice,
    required this.isTrustedLocation,
  });

  Map<String, dynamic> toJson() => {
        "session_id": sessionId,
        "user_id": userId,
        "device_id": deviceId,
        "phone": phone,
        "device_type": deviceType,
        "device_model": deviceModel,
        "os_version": osVersion,
        "app_version": appVersion,
        "network_type": networkType,
        "location_data": locationData,
        "is_known_device": isKnownDevice,
        "is_trusted_location": isTrustedLocation,
      };

  static Future<SessionStartRequest> build({
    required String userId,
    required String phone,
    required bool isKnownDevice,
    required bool isTrustedLocation,
  }) async {
    final deviceInfo = DeviceInfoPlugin();
    final packageInfo = await PackageInfo.fromPlatform();
    final connectivity = await Connectivity().checkConnectivity();
    final position = await Geolocator.getCurrentPosition();

    String deviceType = "mobile";
    String deviceModel = "unknown";
    String osVersion = "unknown";
    String deviceId = "unknown";

    if (defaultTargetPlatform == TargetPlatform.android) {
      final androidInfo = await deviceInfo.androidInfo;
      deviceModel = androidInfo.model ?? "unknown";
      osVersion = androidInfo.version.release ?? "unknown";
      deviceId = androidInfo.id ?? "unknown";
    } else if (defaultTargetPlatform == TargetPlatform.iOS) {
      final iosInfo = await deviceInfo.iosInfo;
      deviceModel = iosInfo.utsname.machine ?? "unknown";
      osVersion = iosInfo.systemVersion ?? "unknown";
      deviceId = iosInfo.identifierForVendor ?? "unknown";
    }

    String networkType = connectivity == ConnectivityResult.wifi
        ? "WiFi"
        : connectivity == ConnectivityResult.mobile
            ? "Mobile"
            : "Unknown";

    return SessionStartRequest(
      sessionId: Uuid().v4(),
      userId: userId,
      deviceId: deviceId,
      phone: phone,
      deviceType: deviceType,
      deviceModel: deviceModel,
      osVersion: osVersion,
      appVersion: packageInfo.version,
      networkType: networkType,
      locationData: {
        "latitude": position.latitude,
        "longitude": position.longitude,
      },
      isKnownDevice: isKnownDevice,
      isTrustedLocation: isTrustedLocation,
    );
  }
} 