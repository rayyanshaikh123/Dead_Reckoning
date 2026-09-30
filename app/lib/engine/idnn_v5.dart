import 'dart:convert';
import 'dart:math' as math;
import 'dart:typed_data';

import 'features_v5.dart';
import 'numeric.dart';

class _Dense {
  _Dense(this.weight, this.bias, this.inDim, this.outDim);

  /// Row-major [outDim × inDim].
  final Float64List weight;
  final Float64List bias;
  final int inDim;
  final int outDim;

  void apply(Float64List x, Float64List out, {required bool activate}) {
    for (var o = 0; o < outDim; o++) {
      var acc = bias[o];
      final row = o * inDim;
      for (var i = 0; i < inDim; i++) {
        acc += weight[row + i] * x[i];
      }
      out[o] = activate ? gelu(acc) : acc;
    }
  }
}

/// IDNN v5 speed model, evaluated in pure Dart.
///
/// Loads the blob written by `tools/app_export/export_model.py`
/// (BatchNorm already folded into the dense layers):
///   input 378 → 256 → 128 → 64 → 1, GELU between layers.
class IdnnV5 {
  IdnnV5._({
    required this.delayTaps,
    required this.nFeatures,
    required Float64List xMean,
    required this._xStd,
    required this.yMean,
    required this.yStd,
    required List<_Dense> layers,
    required this.parameterCount,
  }) : _xMean = xMean,
       _layers = layers,
       _scratch = [
         Float64List(xMean.length),
         for (final l in layers) Float64List(l.outDim),
       ];

  /// Builds the model from the `.json` manifest and `.bin` weight blob.
  factory IdnnV5.fromBytes({
    required String manifestJson,
    required ByteData weights,
  }) {
    final meta = jsonDecode(manifestJson) as Map<String, dynamic>;

    final tensors = <String, (List<int>, Float64List)>{};
    for (final t in (meta['tensors'] as List).cast<Map<String, dynamic>>()) {
      final offset = t['offset'] as int;
      final count = t['count'] as int;
      final data = Float64List(count);
      // getFloat32 is alignment-safe (asset buffers may not be 4-byte aligned).
      for (var i = 0; i < count; i++) {
        data[i] = weights.getFloat32((offset + i) * 4, Endian.little);
      }
      tensors[t['name'] as String] = ((t['shape'] as List).cast<int>(), data);
    }

    _Dense dense(String name) {
      final (shape, w) = tensors['$name.weight']!;
      return _Dense(w, tensors['$name.bias']!.$2, shape[1], shape[0]);
    }

    return IdnnV5._(
      delayTaps: meta['delay_taps'] as int,
      nFeatures: meta['n_features'] as int,
      xMean: tensors['x_mean']!.$2,
      xStd: tensors['x_std']!.$2,
      yMean: (meta['y_mean'] as num).toDouble(),
      yStd: (meta['y_std'] as num).toDouble(),
      layers: [
        dense('in_proj'),
        dense('block1'),
        dense('block2'),
        dense('head'),
      ],
      parameterCount: meta['parameter_count'] as int,
    );
  }

  final int delayTaps;
  final int nFeatures;
  final double yMean;
  final double yStd;
  final int parameterCount;
  final Float64List _xMean;
  final Float64List _xStd;
  final List<_Dense> _layers;
  final List<Float64List> _scratch;

  /// Window length in samples (taps + current).
  int get windowLength => delayTaps + 1;
  int get inputSize => _xMean.length;

  /// Speed (m/s, ≥ 0) for one flattened, *un-normalised* window
  /// (newest sample first, [nFeatures] values per sample).
  double predict(Float64List window) {
    final x = _scratch[0];
    for (var i = 0; i < x.length; i++) {
      x[i] = f32((window[i] - _xMean[i]) / _xStd[i]);
    }
    var h = x;
    for (var l = 0; l < _layers.length; l++) {
      final out = _scratch[l + 1];
      _layers[l].apply(h, out, activate: l < _layers.length - 1);
      h = out;
    }
    return math.max(h[0] * yStd + yMean, 0.0);
  }
}

/// Keeps the last `delayTaps + 1` feature frames and runs the model once
/// the window is full. Mirrors `create_delay_windows_v5` + `predict_v5`:
/// samples before the window fills get speed 0.
class SpeedEstimator {
  SpeedEstimator(this.model)
    : _ring = List.filled(model.windowLength, null),
      _window = Float64List(model.inputSize);

  final IdnnV5 model;
  final List<FeatureFrame?> _ring;
  final Float64List _window;
  int _seen = 0;

  /// Raw neural speed for [f] (m/s).
  double push(FeatureFrame f) {
    final n = _ring.length;
    _ring[_seen % n] = f;
    _seen++;
    if (_seen < n) return 0.0;
    final nf = model.nFeatures;
    for (var tap = 0; tap < n; tap++) {
      final frame = _ring[(_seen - 1 - tap) % n]!;
      _window.setRange(tap * nf, (tap + 1) * nf, frame.values);
    }
    return model.predict(_window);
  }

  void reset() {
    _seen = 0;
    _ring.fillRange(0, _ring.length, null);
  }
}
