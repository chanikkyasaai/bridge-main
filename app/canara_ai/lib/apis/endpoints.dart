class Endpoints {
  static const String baseUrl = 'http://35.225.176.106:8000/api/v1';

  static const String login = "/auth/login";
  static const String register = "/auth/register";
  static const String logout = "/auth/logout";
  static const String verifympin = "/auth/verify-mpin";

  static const String log_start = "/log/start-session";
  static const String log_session = "/log/behavior-data";
  static const String log_end = "/log/end-session";
  static const String log_exit = "/log/app-close";
}
