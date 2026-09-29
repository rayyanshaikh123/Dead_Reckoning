import CoreMotion
import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  private let sensorStream = SensorStreamHandler()

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
    if let registrar = engineBridge.pluginRegistry.registrar(forPlugin: "IdrSensorStream") {
      FlutterEventChannel(name: "idr/sensors", binaryMessenger: registrar.messenger())
        .setStreamHandler(sensorStream)
    }
  }
}

/// Streams device motion to Dart in IDR's canonical frame (Android
/// conventions): m/s², acceleration *including* gravity, gravity vector
/// pointing up (face-up phone ≈ [0, 0, +9.81]), angular rate in rad/s.
///
/// CoreMotion reports in g with the opposite sign (face-up gravity = −1 z),
/// so accelerometer and gravity are negated and scaled; the axes and the
/// gyro convention already match Android.
///
/// Each event: [timestamp s, ax, ay, az, gravX, gravY, gravZ, gyroX, gyroY, gyroZ].
final class SensorStreamHandler: NSObject, FlutterStreamHandler {
  private let motion = CMMotionManager()
  private let queue: OperationQueue = {
    let q = OperationQueue()
    q.maxConcurrentOperationCount = 1
    q.name = "idr.sensors"
    return q
  }()

  func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
    guard motion.isDeviceMotionAvailable else {
      return FlutterError(code: "unavailable", message: "Device motion is not available on this device", details: nil)
    }
    let hz = ((arguments as? [String: Any])?["hz"] as? Double) ?? 50
    motion.deviceMotionUpdateInterval = 1.0 / hz
    motion.startDeviceMotionUpdates(to: queue) { data, error in
      if let error = error {
        DispatchQueue.main.async {
          events(FlutterError(code: "motion", message: error.localizedDescription, details: nil))
        }
        return
      }
      guard let m = data else { return }
      let g = 9.80665
      let grav = m.gravity, user = m.userAcceleration, rate = m.rotationRate
      let sample: [Double] = [
        m.timestamp,
        -(grav.x + user.x) * g, -(grav.y + user.y) * g, -(grav.z + user.z) * g,
        -grav.x * g, -grav.y * g, -grav.z * g,
        rate.x, rate.y, rate.z,
      ]
      DispatchQueue.main.async { events(sample) }
    }
    return nil
  }

  func onCancel(withArguments arguments: Any?) -> FlutterError? {
    motion.stopDeviceMotionUpdates()
    return nil
  }
}
