import 'dart:math' as math;
import 'dart:typed_data';

/// Numeric helpers that reproduce numpy / pandas behaviour exactly where the
/// Python pipeline depends on it.

final _f32 = Float32List(1);

/// Round a double to float32 precision (numpy `.astype(np.float32)`).
double f32(double v) {
  _f32[0] = v;
  return _f32[0];
}

/// Error function, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7).
double erf(double x) {
  final sign = x < 0 ? -1.0 : 1.0;
  final ax = x.abs();
  final t = 1.0 / (1.0 + 0.3275911 * ax);
  final y =
      1.0 -
      (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t -
                      0.284496736) *
                  t +
              0.254829592) *
          t *
          math.exp(-ax * ax);
  return sign * y;
}

/// Exact (erf-based) GELU, as in `torch.nn.GELU()`.
double gelu(double x) => 0.5 * x * (1.0 + erf(x / math.sqrt2));

/// pandas `Series.rolling(window, min_periods=1).var().fillna(0.0)`
/// (sample variance, ddof = 1; a single value gives NaN → 0).
class RollingVariance {
  RollingVariance(this.window) : _buf = Float64List(window);

  final int window;
  final Float64List _buf;
  int _count = 0;
  int _head = 0;

  double push(double v) {
    _buf[_head] = v;
    _head = (_head + 1) % window;
    if (_count < window) _count++;
    if (_count < 2) return 0.0;
    var mean = 0.0;
    for (var i = 0; i < _count; i++) {
      mean += _buf[i];
    }
    mean /= _count;
    var ss = 0.0;
    for (var i = 0; i < _count; i++) {
      final d = _buf[i] - mean;
      ss += d * d;
    }
    return ss / (_count - 1);
  }

  void reset() {
    _count = 0;
    _head = 0;
  }
}

/// numpy `np.percentile(values, q)` with the default linear interpolation.
double percentile(List<double> values, double q) {
  if (values.isEmpty) return double.nan;
  final s = [...values]..sort();
  final pos = (s.length - 1) * q / 100.0;
  final lo = pos.floor();
  final hi = pos.ceil();
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

/// numpy `np.median`.
double median(List<double> values) => percentile(values, 50);
