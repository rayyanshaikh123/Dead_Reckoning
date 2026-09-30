import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart' show Distance, LatLng, LengthUnit;

import '../theme/theme.dart';

/// Google-Maps-style "you are here" indicator for a [FlutterMap]:
/// accuracy circle (metres), heading beam, white-ringed dot with a soft
/// pulse. Position and heading glide between updates instead of jumping.
class LocationPuckLayer extends StatefulWidget {
  const LocationPuckLayer({
    super.key,
    required this.position,
    this.headingDeg,
    this.accuracyM,
    this.color = IdrColors.location,
    this.stale = false,
    this.onMoved,
  });

  final LatLng position;

  /// Direction of travel, degrees clockwise from north (null = no beam).
  final double? headingDeg;

  /// Radius of the uncertainty circle in metres (null = none).
  final double? accuracyM;
  final Color color;

  /// Fix is old: drawn grey, no beam, no pulse.
  final bool stale;

  /// Called with the on-screen (animated) position every frame it moves,
  /// e.g. to keep the camera following smoothly.
  final ValueChanged<LatLng>? onMoved;

  @override
  State<LocationPuckLayer> createState() => _LocationPuckLayerState();
}

class _LocationPuckLayerState extends State<LocationPuckLayer>
    with TickerProviderStateMixin {
  late final AnimationController _move = AnimationController(
    vsync: this,
    // Slightly shorter than the 1 s GPS interval so the dot settles.
    duration: const Duration(milliseconds: 900),
  )..addListener(_onMove);
  late final AnimationController _pulse = AnimationController(
    vsync: this,
    duration: const Duration(milliseconds: 2200),
  )..repeat();

  late LatLng _from = widget.position, _to = widget.position;
  double? _hFrom, _hTo;
  double? _accFrom, _accTo;

  @override
  void initState() {
    super.initState();
    _hFrom = _hTo = widget.headingDeg;
    _accFrom = _accTo = widget.accuracyM;
  }

  @override
  void didUpdateWidget(LocationPuckLayer old) {
    super.didUpdateWidget(old);
    if (widget.position != old.position ||
        widget.headingDeg != old.headingDeg ||
        widget.accuracyM != old.accuracyM) {
      _from = _position;
      _hFrom = _heading;
      _accFrom = _accuracy;
      _to = widget.position;
      _hTo = widget.headingDeg;
      _accTo = widget.accuracyM;
      // A big jump (first fix, re-centre after an outage) snaps instead.
      final far = const Distance().as(LengthUnit.Meter, _from, _to) > 300;
      _move.forward(from: far ? 1 : 0);
    }
  }

  double get _t => Curves.easeOut.transform(_move.value);

  LatLng get _position => LatLng(
    _from.latitude + (_to.latitude - _from.latitude) * _t,
    _from.longitude + (_to.longitude - _from.longitude) * _t,
  );

  double? get _heading {
    final a = _hFrom, b = _hTo;
    if (b == null) return null;
    if (a == null) return b;
    // Turn the short way round.
    final d = ((b - a + 540) % 360) - 180;
    return (a + d * _t) % 360;
  }

  double? get _accuracy {
    final a = _accFrom, b = _accTo;
    if (b == null) return null;
    if (a == null) return b;
    return a + (b - a) * _t;
  }

  void _onMove() => widget.onMoved?.call(_position);

  @override
  void dispose() {
    _move.dispose();
    _pulse.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: Listenable.merge([_move, _pulse]),
      builder: (context, _) {
        final p = _position;
        final acc = _accuracy;
        final color = widget.stale ? IdrColors.textMuted : widget.color;
        return Stack(
          children: [
            if (acc != null && acc > 0)
              CircleLayer(
                circles: [
                  CircleMarker(
                    point: p,
                    radius: acc,
                    useRadiusInMeter: true,
                    color: color.withValues(alpha: 0.14),
                    borderColor: color.withValues(alpha: 0.35),
                    borderStrokeWidth: 1,
                  ),
                ],
              ),
            MarkerLayer(
              markers: [
                Marker(
                  point: p,
                  width: 150,
                  height: 150,
                  child: IgnorePointer(
                    child: CustomPaint(
                      painter: _PuckPainter(
                        color: color,
                        headingDeg: widget.stale ? null : _heading,
                        pulse: widget.stale ? null : _pulse.value,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ],
        );
      },
    );
  }
}

class _PuckPainter extends CustomPainter {
  _PuckPainter({
    required this.color,
    required this.headingDeg,
    required this.pulse,
  });

  final Color color;
  final double? headingDeg;
  final double? pulse;

  static const _dot = 8.5;
  static const _ring = 3.0;

  @override
  void paint(Canvas canvas, Size size) {
    final c = size.center(Offset.zero);

    // Heading beam: a soft cone fading outwards, like Google's "flashlight".
    final h = headingDeg;
    if (h != null) {
      const spread = 32 * math.pi / 180;
      const reach = 58.0;
      canvas.save();
      canvas.translate(c.dx, c.dy);
      canvas.rotate(h * math.pi / 180);
      final beam = Path()
        ..moveTo(0, 0)
        ..arcTo(
          Rect.fromCircle(center: Offset.zero, radius: reach),
          -math.pi / 2 - spread,
          2 * spread,
          false,
        )
        ..close();
      canvas.drawPath(
        beam,
        Paint()
          ..shader = RadialGradient(
            colors: [color.withValues(alpha: 0.55), color.withValues(alpha: 0)],
          ).createShader(Rect.fromCircle(center: Offset.zero, radius: reach)),
      );
      canvas.restore();
    }

    // Breathing halo.
    final p = pulse;
    if (p != null) {
      canvas.drawCircle(
        c,
        _dot + _ring + 4 + p * 14,
        Paint()..color = color.withValues(alpha: 0.28 * (1 - p)),
      );
    }

    // Shadow, white ring, coloured dot.
    canvas.drawCircle(
      c.translate(0, 1),
      _dot + _ring + 1,
      Paint()
        ..color = Colors.black.withValues(alpha: 0.45)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 3),
    );
    canvas.drawCircle(c, _dot + _ring, Paint()..color = Colors.white);
    canvas.drawCircle(c, _dot, Paint()..color = color);
  }

  @override
  bool shouldRepaint(_PuckPainter old) =>
      old.color != color || old.headingDeg != headingDeg || old.pulse != pulse;
}

/// Round "my location" button, highlighted while the map is not following.
class RecenterButton extends StatelessWidget {
  const RecenterButton({
    super.key,
    required this.following,
    required this.onTap,
  });

  final bool following;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: following ? 'Following your location' : 'Centre on my location',
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          width: 48,
          height: 48,
          decoration: BoxDecoration(
            color: IdrColors.surface,
            shape: BoxShape.circle,
            border: Border.all(color: IdrColors.outline),
            boxShadow: const [
              BoxShadow(
                color: Color(0x66000000),
                blurRadius: 10,
                offset: Offset(0, 3),
              ),
            ],
          ),
          child: Icon(
            following ? Icons.my_location : Icons.location_searching,
            size: 22,
            color: following ? IdrColors.location : IdrColors.textPrimary,
          ),
        ),
      ),
    );
  }
}
