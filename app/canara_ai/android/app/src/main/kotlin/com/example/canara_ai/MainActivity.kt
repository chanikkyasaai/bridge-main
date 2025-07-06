package com.example.canara_ai

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import com.example.canara_ai.input.InputSensorPlugin

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        // Correct way to manually register the plugin
        flutterEngine.plugins.add(InputSensorPlugin())
    }
}
