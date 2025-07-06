package com.example.canara_ai.input

import android.app.Activity
import android.view.MotionEvent
import android.view.View
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler

class InputSensorPlugin : FlutterPlugin, MethodCallHandler, ActivityAware {
    private lateinit var channel: MethodChannel
    private var lastMotionEvent: MotionEvent? = null
    private var activity: Activity? = null

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(binding.binaryMessenger, "input_sensor")
        channel.setMethodCallHandler(this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
    }

    override fun onAttachedToActivity(binding: ActivityPluginBinding) {
        activity = binding.activity

        // Attach listener to root view
        activity?.window?.decorView?.rootView?.setOnTouchListener { _, event ->
            lastMotionEvent = event
            false
        }
    }

    override fun onDetachedFromActivityForConfigChanges() {}
    override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) {}
    override fun onDetachedFromActivity() {}

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        if (call.method == "getTouchInfo") {
            val event = lastMotionEvent
            if (event != null) {
                result.success(
                    mapOf(
                        "pressure" to event.pressure,
                        "orientation" to event.orientation,
                        "size" to event.size
                    )
                )
            } else {
                result.error("NO_EVENT", "No MotionEvent recorded", null)
            }
        } else {
            result.notImplemented()
        }
    }
}
