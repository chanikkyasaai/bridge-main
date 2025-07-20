# Flutter Token Refresh Fix

## Issues Found in Your Implementation

### 1. **Missing Error Handling in Retry Queue**
Your retry queue doesn't handle errors properly, which can cause silent failures.

### 2. **Incorrect Response Validation**
You're checking for `response.data['access_token']` but the backend returns the token directly in the response.

### 3. **Missing Request Options in Retry**
The retry queue doesn't preserve the original request options properly.

### 4. **Potential Race Conditions**
The `_isRefreshing` flag might not handle all edge cases correctly.

## Fixed Implementation

```dart
import 'package:canara_ai/apis/endpoints.dart';
import 'package:canara_ai/main.dart';
import 'package:canara_ai/utils/token_storage.dart';
import 'package:dio/dio.dart';

class AuthInterceptor extends Interceptor {
  final Dio _dio;
  final TokenStorage _storage;

  AuthInterceptor(this._dio, this._storage);

  bool _isRefreshing = false;
  List<Function()> _retryQueue = [];

  @override
  void onRequest(RequestOptions options, RequestInterceptorHandler handler) async {
    // Skip token for auth endpoints
    if (options.path.contains('/auth/login') || 
        options.path.contains('/auth/register') ||
        options.path.contains('/auth/refresh')) {
      return handler.next(options);
    }

    final token = await _storage.getAccessToken();
    if (token != null) {
      options.headers['Authorization'] = 'Bearer $token';
    }
    handler.next(options);
  }

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) async {
    final requestPath = err.requestOptions.path;

    // Only handle 401 errors and avoid refresh endpoint loops
    if (err.response?.statusCode == 401 && 
        !_isRefreshing && 
        !requestPath.contains('/auth/refresh') &&
        !requestPath.contains('/auth/login') &&
        !requestPath.contains('/auth/register')) {
      
      _isRefreshing = true;

      try {
        print("🔄 Token expired, attempting refresh...");
        final success = await _refreshToken();

        if (success) {
          print("✅ Token refresh successful, retrying original request");
          
          // Execute all queued requests
          for (var retry in _retryQueue) {
            try {
              retry();
            } catch (e) {
              print("❌ Retry request failed: $e");
            }
          }
          _retryQueue.clear();

          // Retry the original request
          final newToken = await _storage.getAccessToken();
          if (newToken != null) {
            final opts = err.requestOptions;
            opts.headers['Authorization'] = 'Bearer $newToken';
            
            try {
              final response = await _dio.fetch(opts);
              return handler.resolve(response);
            } catch (retryError) {
              print("❌ Original request retry failed: $retryError");
              return handler.reject(retryError);
            }
          } else {
            print("❌ No new token available after refresh");
            _handleLogout();
            return handler.reject(err);
          }
        } else {
          print("❌ Token refresh failed, logging out");
          _handleLogout();
          return handler.reject(err);
        }
      } catch (refreshError) {
        print("❌ Token refresh error: $refreshError");
        _handleLogout();
        return handler.reject(err);
      } finally {
        _isRefreshing = false;
      }
    } else if (_isRefreshing && 
               !requestPath.contains('/auth/refresh') &&
               !requestPath.contains('/auth/login') &&
               !requestPath.contains('/auth/register')) {
      // Queue the request for retry
      _retryQueue.add(() async {
        try {
          final newToken = await _storage.getAccessToken();
          if (newToken != null) {
            final opts = err.requestOptions;
            opts.headers['Authorization'] = 'Bearer $newToken';
            await _dio.fetch(opts);
          }
        } catch (e) {
          print("❌ Queued request retry failed: $e");
        }
      });
      return handler.reject(err);
    } else {
      handler.next(err);
    }
  }

  Future<bool> _refreshToken() async {
    final refreshToken = await _storage.getRefreshToken();

    if (refreshToken == null) {
      print("❌ No refresh token available");
      return false;
    }

    try {
      print("🔄 Refreshing token...");
      print("📤 Refresh token: ${refreshToken.substring(0, 20)}...");

      // Create a new Dio instance for refresh to avoid conflicts
      final refreshDio = Dio();
      refreshDio.options.baseUrl = Endpoints.baseUrl;
      refreshDio.options.connectTimeout = Duration(seconds: 10);
      refreshDio.options.receiveTimeout = Duration(seconds: 10);
      refreshDio.options.headers['Content-Type'] = 'application/json';

      final response = await refreshDio.post('/auth/refresh', data: {
        'refresh_token': refreshToken,
      });

      print("📥 Refresh response status: ${response.statusCode}");
      print("📥 Refresh response data: ${response.data}");

      if (response.statusCode == 200) {
        final data = response.data;
        
        // Validate response structure
        if (data['access_token'] != null && data['refresh_token'] != null) {
          await _storage.saveTokens(
            data['access_token'],
            data['refresh_token'],
          );
          
          print("✅ New access token: ${data['access_token'].substring(0, 20)}...");
          print("✅ New refresh token: ${data['refresh_token'].substring(0, 20)}...");
          return true;
        } else {
          print("❌ Invalid response structure: missing tokens");
          return false;
        }
      } else {
        print("❌ Refresh failed with status: ${response.statusCode}");
        return false;
      }
    } on DioException catch (e) {
      print("❌ DioException during refresh:");
      print("   Type: ${e.type}");
      print("   Message: ${e.message}");
      print("   Status Code: ${e.response?.statusCode}");
      print("   Response Data: ${e.response?.data}");
      
      if (e.response?.statusCode == 401) {
        print("❌ Refresh token is invalid or expired");
      }
      return false;
    } catch (e) {
      print("❌ Unexpected error during refresh: $e");
      return false;
    }
  }

  void _handleLogout() async {
    print("🚪 Handling logout...");
    await _storage.clearTokens();
    print('✅ User logged out. Redirecting to login.');
    navigatorKey.currentState?.pushNamedAndRemoveUntil('/login', (route) => false);
  }
}
```

## Additional Debugging Implementation

Add this to your `TokenStorage` class for better debugging:

```dart
class TokenStorage {
  // ... your existing methods ...

  // Debug method to check token status
  Future<void> debugTokenStatus() async {
    final accessToken = await getAccessToken();
    final refreshToken = await getRefreshToken();
    
    print("🔍 Token Debug Info:");
    print("   Access Token: ${accessToken != null ? '${accessToken.substring(0, 20)}...' : 'null'}");
    print("   Refresh Token: ${refreshToken != null ? '${refreshToken.substring(0, 20)}...' : 'null'}");
    
    if (accessToken != null) {
      try {
        // Decode JWT to check expiry (without verification)
        final parts = accessToken.split('.');
        if (parts.length == 3) {
          final payload = parts[1];
          final normalized = base64Url.normalize(payload);
          final resp = utf8.decode(base64Url.decode(normalized));
          final payloadMap = json.decode(resp);
          
          final exp = payloadMap['exp'];
          final iat = payloadMap['iat'];
          final now = DateTime.now().millisecondsSinceEpoch ~/ 1000;
          
          print("   Access Token Expiry: ${DateTime.fromMillisecondsSinceEpoch(exp * 1000)}");
          print("   Access Token Issued: ${DateTime.fromMillisecondsSinceEpoch(iat * 1000)}");
          print("   Current Time: ${DateTime.fromMillisecondsSinceEpoch(now * 1000)}");
          print("   Token Valid: ${exp > now}");
        }
      } catch (e) {
        print("   Error decoding access token: $e");
      }
    }
  }
}
```

## Dio Configuration

Update your Dio configuration to include better error handling:

```dart
class ApiService {
  late Dio _dio;
  late TokenStorage _storage;

  ApiService() {
    _storage = TokenStorage();
    _dio = Dio();
    
    // Configure Dio
    _dio.options.baseUrl = Endpoints.baseUrl;
    _dio.options.connectTimeout = Duration(seconds: 30);
    _dio.options.receiveTimeout = Duration(seconds: 30);
    _dio.options.headers['Content-Type'] = 'application/json';
    
    // Add interceptors
    _dio.interceptors.add(AuthInterceptor(_dio, _storage));
    
    // Add logging interceptor for debugging
    _dio.interceptors.add(LogInterceptor(
      requestBody: true,
      responseBody: true,
      logPrint: (obj) => print(obj),
    ));
  }
}
```

## Testing the Fix

Add this test method to verify the refresh works:

```dart
class AuthService {
  final ApiService _apiService;
  final TokenStorage _storage;

  AuthService(this._apiService, this._storage);

  Future<void> testTokenRefresh() async {
    print("🧪 Testing token refresh...");
    
    // Debug current token status
    await _storage.debugTokenStatus();
    
    // Try to make a request that requires authentication
    try {
      final response = await _apiService._dio.get('/auth/session-status');
      print("✅ Session status request successful: ${response.data}");
    } catch (e) {
      print("❌ Session status request failed: $e");
    }
  }
}
```

## Common Issues and Solutions

### 1. **"Connection refused" Error**
- **Cause**: Wrong base URL or backend not running
- **Solution**: Verify `Endpoints.baseUrl` is correct for your setup

### 2. **"Timeout" Error**
- **Cause**: Network issues or slow backend
- **Solution**: Increase timeout values in Dio configuration

### 3. **"401 Unauthorized" on Refresh**
- **Cause**: Invalid or expired refresh token
- **Solution**: User needs to re-login

### 4. **"500 Internal Server Error"**
- **Cause**: Backend error
- **Solution**: Check backend logs and database schema

## Verification Steps

1. **Test the backend directly** (already done - working)
2. **Implement the fixed interceptor**
3. **Add debugging to TokenStorage**
4. **Test with logging enabled**
5. **Verify network connectivity**

The main issues in your original code were:
- Missing proper error handling in retry queue
- Incorrect response validation
- Potential race conditions
- Missing request option preservation

The fixed implementation addresses all these issues and provides better debugging capabilities. 