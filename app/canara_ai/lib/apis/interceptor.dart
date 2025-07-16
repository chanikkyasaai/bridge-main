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
    final token = await _storage.getAccessToken();
    if (token != null) {
      options.headers['Authorization'] = 'Bearer $token';
    }
    handler.next(options);
  }

  @override
  void onError(DioException err, ErrorInterceptorHandler handler) async {
    final requestPath = err.requestOptions.path;

    if (err.response?.statusCode == 401 && !_isRefreshing && !requestPath.contains('/auth/refresh')) {
      _isRefreshing = true;

      try {
        final success = await _refreshToken();

        if (success) {
          for (var retry in _retryQueue) {
            retry();
          }
          _retryQueue.clear();

          final newToken = await _storage.getAccessToken();
          final opts = err.requestOptions;
          opts.headers['Authorization'] = 'Bearer $newToken';
          final response = await _dio.fetch(opts);
          return handler.resolve(response);
        } else {
          _handleLogout();
          return handler.reject(err);
        }
      } catch (_) {
        _handleLogout();
        return handler.reject(err);
      } finally {
        _isRefreshing = false;
      }
    } else if (_isRefreshing) {
      _retryQueue.add(() {
        final opts = err.requestOptions;
        _dio.fetch(opts);
      });
    } else {
      handler.next(err);
    }
  }

  Future<bool> _refreshToken() async {
    final refreshToken = await _storage.getRefreshToken();

    if (refreshToken == null) return false;

    try {
      print("Refreshing token with refresh token: $refreshToken");

      _dio.options.headers['Content-Type'] = 'application/json';
      final response = await _dio.post('${Endpoints.baseUrl}/auth/refresh', data: {
        'refresh_token': refreshToken,
      }).timeout(Duration(seconds: 10));
      ;

      print("Refresh token: ${response.data['access_token']}");

      if (response.statusCode == 200 && response.data['access_token'] != null && response.data['refresh_token'] != null) {
        await _storage.saveTokens(
          response.data['access_token'],
          response.data['refresh_token'],
        );
        return true;
      }
      else if(response.statusCode == 401) {
        return false;
      }
    } catch (_) {
      print("Failed to refresh token, logging out.");
    }
    return false;
  }

  void _handleLogout() async {
    await _storage.clearTokens();
    print('User logged out. Redirect to login.');
    navigatorKey.currentState?.pushNamedAndRemoveUntil('/login', (route) => false);
  }
}
