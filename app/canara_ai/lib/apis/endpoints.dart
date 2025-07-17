class Endpoints {
  static const String baseUrl = 'http://192.168.241.41:8000/api/v1';

  static const String login = "/auth/login";
  static const String register = "/auth/register";
  static const String logout = "/auth/logout";
  static const String verifympin = "/auth/verify-mpin";

  static const String log_start = "/log/start-session";
  static const String log_session = "/log/behavior-data";
  static const String log_end = "/log/end-session";
  static const String log_exit = "/log/app-close";

  // ML Engine Endpoints
  static const String ml_status = "/ml/ml-engine/status";
  static const String ml_session_info = "/ml/ml-engine/session/"; // append {session_id}
  static const String ml_alerts = "/ml/ml-alerts";
  static const String ml_behavior_summary = "/ml/sessions/"; // append {session_id}/behavior-summary
  static const String ml_simulate_analysis = "/ml/sessions/"; // append {session_id}/simulate-ml-analysis
}
