import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';

/// Appends uncaught errors to `errors.log` on the phone so problems from a
/// test drive can be shared (Settings → Report a problem). Nothing is sent
/// anywhere automatically.
class ErrorLog {
  ErrorLog._();

  static const _maxBytes = 256 * 1024;
  static Future<File>? _file;

  static Future<File> file() => _file ??= getApplicationSupportDirectory().then(
    (d) => File('${d.path}/errors.log'),
  );

  /// Hooks Flutter and platform error handlers. Call once from `main()`.
  static void install() {
    final previous = FlutterError.onError;
    FlutterError.onError = (details) {
      record(
        details.exception,
        details.stack,
        context: details.context?.toString(),
      );
      previous?.call(details);
    };
    PlatformDispatcher.instance.onError = (error, stack) {
      record(error, stack);
      return true; // handled: don't crash the app for a stray async error
    };
  }

  static Future<void> record(
    Object error,
    StackTrace? stack, {
    String? context,
  }) async {
    debugPrint('IDR error: $error');
    try {
      final f = await file();
      if (await f.exists() && await f.length() > _maxBytes) {
        // Keep the newest half.
        final text = await f.readAsString();
        await f.writeAsString(text.substring(text.length ~/ 2));
      }
      final lines = StringBuffer()
        ..writeln(
          '--- ${DateTime.now().toIso8601String()}${context == null ? '' : ' ($context)'}',
        )
        ..writeln(error)
        ..writeln(stack?.toString().split('\n').take(12).join('\n') ?? '');
      await f.writeAsString(lines.toString(), mode: FileMode.append);
    } catch (_) {
      // Logging must never throw.
    }
  }

  static Future<bool> hasEntries() async {
    try {
      final f = await file();
      return await f.exists() && await f.length() > 0;
    } catch (_) {
      return false;
    }
  }

  static Future<void> clear() async {
    try {
      final f = await file();
      if (await f.exists()) await f.delete();
    } catch (_) {}
  }
}
