import 'package:device_info_plus/device_info_plus.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:geolocator/geolocator.dart';
import 'package:flutter/foundation.dart';

class SessionStartRequest {
  final String mpin;
  final String deviceId;
  final String deviceType;
  final String deviceModel;
  final String osVersion;
  final String appVersion;
  final String networkType;
  final Map<String, dynamic> locationData;
  final bool isKnownDevice;
  final bool isTrustedLocation;
  final String userAgent;

  SessionStartRequest({
    required this.mpin,
    required this.deviceId,
    required this.deviceType,
    required this.deviceModel,
    required this.osVersion,
    required this.appVersion,
    required this.networkType,
    required this.locationData,
    required this.isKnownDevice,
    required this.isTrustedLocation,
    required this.userAgent,
  });

  Map<String, dynamic> toJson() => {
        "mpin": mpin,
        "context": {
          "device_id": deviceId,
          "device_type": deviceType,
          "device_model": deviceModel,
          "os_version": osVersion,
          "app_version": appVersion,
          "network_type": networkType,
          "location_data": locationData,
          "is_known_device": isKnownDevice,
          "is_trusted_location": isTrustedLocation,
          "user_agent": userAgent
        }
      };

  static Future<SessionStartRequest> build({
    required String mpin,
    required bool isKnownDevice,
    required bool isTrustedLocation,
  }) async {
    final deviceInfo = DeviceInfoPlugin();
    final packageInfo = await PackageInfo.fromPlatform();
    final connectivity = await Connectivity().checkConnectivity();

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

    Map<String, dynamic> locationData = {
      "latitude": null,
      "longitude": null,
    };

    try {
      bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
      if (!serviceEnabled) {
        throw Exception('Location services are disabled.');
      }

      LocationPermission permission = await Geolocator.checkPermission();
      if (permission == LocationPermission.denied) {
        permission = await Geolocator.requestPermission();
        if (permission == LocationPermission.denied) {
          throw Exception('Location permissions denied by user.');
        }
      }
      if (permission == LocationPermission.deniedForever) {
        throw Exception('Location permissions permanently denied.');
      }

      final position = await Geolocator.getCurrentPosition();
      locationData = {
        "latitude": position.latitude,
        "longitude": position.longitude,
      };
    } catch (e) {
      debugPrint('Location error caught: $e');
      // Optionally: Log this to backend for diagnostics.
      // You can also include `location_disabled: true` in context if needed.
    }

    return SessionStartRequest(
      mpin: mpin,
      deviceId: deviceId,
      deviceType: deviceType,
      deviceModel: deviceModel,
      osVersion: osVersion,
      appVersion: packageInfo.version,
      networkType: networkType,
      locationData: locationData,
      isKnownDevice: isKnownDevice,
      isTrustedLocation: isTrustedLocation,
      userAgent: 'CanaraAI1/${packageInfo.version} ($deviceType; $deviceModel; Android $osVersion)',
    );
  }
}
