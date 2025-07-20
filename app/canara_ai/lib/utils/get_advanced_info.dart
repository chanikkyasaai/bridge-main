import 'dart:io';
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
  final String userAgent;
  final String deviceIp;

  SessionStartRequest({
    required this.mpin,
    required this.deviceId,
    required this.deviceType,
    required this.deviceModel,
    required this.osVersion,
    required this.appVersion,
    required this.networkType,
    required this.locationData,
    required this.userAgent,
    required this.deviceIp,
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
          "user_agent": userAgent,
          "ip_address": deviceIp,
        }
      };

  static Future<String> getDeviceIp() async {
    try {
      for (var interface in await NetworkInterface.list()) {
        for (var addr in interface.addresses) {
          if (addr.type == InternetAddressType.IPv4 && !addr.isLoopback) {
            return addr.address;
          }
        }
      }
    } catch (e) {
      debugPrint('Error getting IP address: $e');
    }
    return 'unknown';
  }

  static Future<SessionStartRequest> build({
    required String mpin,
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
    }

    final deviceIp = await getDeviceIp();

    return SessionStartRequest(
      mpin: mpin,
      deviceId: deviceId,
      deviceType: deviceType,
      deviceModel: deviceModel,
      osVersion: osVersion,
      appVersion: packageInfo.version,
      networkType: networkType,
      locationData: locationData,
      userAgent: 'CanaraAI1/${packageInfo.version} ($deviceType; $deviceModel; Android $osVersion)',
      deviceIp: deviceIp,
    );
  }
}
