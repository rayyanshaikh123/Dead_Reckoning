package com.sihidr.idr_app

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        EventChannel(flutterEngine.dartExecutor.binaryMessenger, "idr/sensors")
            .setStreamHandler(SensorStreamHandler(applicationContext))
    }
}

/**
 * Streams motion sensors to Dart in IDR's canonical frame, which is Android's
 * own: m/s², accelerometer including gravity, gravity vector pointing up,
 * gyroscope in rad/s.
 *
 * Each event (emitted per accelerometer reading, carrying the latest gravity
 * and gyro values): [timestamp s, ax, ay, az, gravX, gravY, gravZ, gyroX, gyroY, gyroZ].
 * Gravity entries are NaN on devices without a gravity sensor.
 */
private class SensorStreamHandler(context: Context) : EventChannel.StreamHandler, SensorEventListener {
    private val manager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private var sink: EventChannel.EventSink? = null
    private val gravity = floatArrayOf(Float.NaN, Float.NaN, Float.NaN)
    private val gyro = floatArrayOf(0f, 0f, 0f)

    override fun onListen(arguments: Any?, events: EventChannel.EventSink) {
        val accel = manager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        if (accel == null) {
            events.error("unavailable", "No accelerometer on this device", null)
            return
        }
        sink = events
        val hz = ((arguments as? Map<*, *>)?.get("hz") as? Number)?.toDouble() ?: 50.0
        val periodUs = (1_000_000 / hz).toInt()
        manager.registerListener(this, accel, periodUs)
        manager.getDefaultSensor(Sensor.TYPE_GRAVITY)?.let { manager.registerListener(this, it, periodUs) }
        manager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)?.let { manager.registerListener(this, it, periodUs) }
    }

    override fun onCancel(arguments: Any?) {
        manager.unregisterListener(this)
        sink = null
    }

    override fun onSensorChanged(event: SensorEvent) {
        when (event.sensor.type) {
            Sensor.TYPE_GRAVITY -> event.values.copyInto(gravity, endIndex = 3)
            Sensor.TYPE_GYROSCOPE -> event.values.copyInto(gyro, endIndex = 3)
            Sensor.TYPE_ACCELEROMETER -> sink?.success(
                doubleArrayOf(
                    event.timestamp / 1e9,
                    event.values[0].toDouble(), event.values[1].toDouble(), event.values[2].toDouble(),
                    gravity[0].toDouble(), gravity[1].toDouble(), gravity[2].toDouble(),
                    gyro[0].toDouble(), gyro[1].toDouble(), gyro[2].toDouble(),
                )
            )
        }
    }

    override fun onAccuracyChanged(sensor: Sensor, accuracy: Int) = Unit
}
