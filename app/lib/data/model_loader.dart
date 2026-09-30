import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../engine/engine.dart';

/// The on-device model plus a quick self-test result.
class ModelStatus {
  const ModelStatus({required this.model, required this.msPerStep});

  final IdnnV5 model;

  /// Average inference time measured on this device.
  final double msPerStep;
}

/// Loads IDNN v5 from the bundled assets and times a few inferences.
final speedModelProvider = FutureProvider<ModelStatus>((ref) async {
  final manifest = await rootBundle.loadString('assets/models/idnn_v5.json');
  final weights = await rootBundle.load('assets/models/idnn_v5.bin');
  final model = IdnnV5.fromBytes(manifestJson: manifest, weights: weights);

  final window = Float64List(model.inputSize);
  model.predict(window); // warm-up
  const runs = 50;
  final sw = Stopwatch()..start();
  for (var i = 0; i < runs; i++) {
    model.predict(window);
  }
  return ModelStatus(
    model: model,
    msPerStep: sw.elapsedMicroseconds / runs / 1000,
  );
});
