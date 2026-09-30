import 'package:flutter/material.dart';

import '../theme/theme.dart';

/// Front three-quarter line drawing of a sedan, bleeding off the right edge
/// the way the dashboard mockup does. Drawn in a 280×510 design box and scaled
/// to the available height.
class CarLineArt extends StatelessWidget {
  const CarLineArt({
    super.key,
    this.color = IdrColors.textPrimary,
    this.headlightsOn = false,
  });

  final Color color;

  /// Fills the headlight lenses with the accent colour (used while DR is active).
  final bool headlightsOn;

  @override
  Widget build(BuildContext context) {
    return ClipRect(
      child: LayoutBuilder(
        builder: (context, box) {
          final scale = box.maxHeight / _h;
          return OverflowBox(
            alignment: Alignment.centerLeft,
            maxWidth: _w * scale,
            minWidth: _w * scale,
            child: CustomPaint(
              size: Size(_w * scale, box.maxHeight),
              painter: _CarPainter(color: color, headlightsOn: headlightsOn),
            ),
          );
        },
      ),
    );
  }
}

const _w = 280.0;
const _h = 512.0;

class _CarPainter extends CustomPainter {
  _CarPainter({required this.color, required this.headlightsOn});

  final Color color;
  final bool headlightsOn;

  @override
  void paint(Canvas canvas, Size size) {
    final s = size.height / _h;
    canvas.scale(s);
    final p = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.6 / s.clamp(0.6, 2.0)
      ..strokeJoin = StrokeJoin.round
      ..strokeCap = StrokeCap.round;

    // Roof and A-pillar.
    canvas.drawPath(
      Path()
        ..moveTo(290, 2)
        ..lineTo(160, 6)
        ..quadraticBezierTo(126, 9, 108, 52)
        ..lineTo(70, 150)
        ..lineTo(62, 176),
      p,
    );

    // Windscreen.
    canvas.drawPath(
      Path()
        ..moveTo(290, 22)
        ..lineTo(172, 24)
        ..quadraticBezierTo(152, 26, 142, 52)
        ..lineTo(104, 150)
        ..lineTo(290, 147),
      p,
    );

    // Wing mirror.
    canvas.drawPath(
      Path()
        ..moveTo(70, 112)
        ..quadraticBezierTo(22, 106, 4, 126)
        ..quadraticBezierTo(-4, 152, 22, 162)
        ..lineTo(66, 164)
        ..quadraticBezierTo(80, 150, 84, 128)
        ..quadraticBezierTo(82, 112, 70, 112),
      p,
    );
    canvas.drawPath(
      Path()
        ..moveTo(84, 128)
        ..lineTo(98, 132)
        ..lineTo(92, 162),
      p,
    );

    // Front fender down to the wheel.
    canvas.drawPath(
      Path()
        ..moveTo(62, 176)
        ..quadraticBezierTo(50, 196, 44, 214)
        ..quadraticBezierTo(22, 228, 20, 250)
        ..lineTo(20, 440),
      p,
    );

    // Hood crease / top of the headlight line.
    canvas.drawPath(
      Path()
        ..moveTo(44, 214)
        ..quadraticBezierTo(150, 170, 290, 168),
      p,
    );
    canvas.drawPath(
      Path()
        ..moveTo(26, 240)
        ..quadraticBezierTo(150, 238, 290, 244),
      p,
    );

    // Headlight housing.
    final lamp = Path()
      ..moveTo(34, 252)
      ..quadraticBezierTo(38, 300, 84, 312)
      ..lineTo(176, 302)
      ..lineTo(184, 276)
      ..quadraticBezierTo(110, 256, 34, 252)
      ..close();
    canvas.drawPath(lamp, p);

    final lensFill = Paint()
      ..color = IdrColors.accent.withValues(alpha: headlightsOn ? 0.9 : 0);
    for (final r in const [
      Rect.fromLTWH(50, 262, 44, 40),
      Rect.fromLTWH(98, 268, 44, 36),
    ]) {
      final rr = RRect.fromRectAndRadius(r, const Radius.circular(18));
      if (headlightsOn) canvas.drawRRect(rr, lensFill);
      canvas.drawRRect(rr, p);
      canvas.drawLine(
        Offset(r.left + 12, r.center.dy),
        Offset(r.right - 12, r.center.dy + 2),
        p,
      );
    }

    // Kidney grille, clipped by the right edge.
    final grille = RRect.fromRectAndRadius(
      const Rect.fromLTWH(196, 262, 110, 72),
      const Radius.circular(18),
    );
    canvas.drawRRect(grille, p);
    canvas.save();
    canvas.clipRRect(grille);
    for (double x = 208; x < 300; x += 9) {
      canvas.drawLine(Offset(x, 262), Offset(x, 334), p);
    }
    canvas.restore();
    canvas.drawRRect(grille.inflate(-6), p);

    // Body line under the lamps.
    canvas.drawPath(
      Path()
        ..moveTo(26, 322)
        ..quadraticBezierTo(150, 350, 290, 342),
      p,
    );

    // Sensors / fog dots.
    canvas.drawCircle(const Offset(54, 338), 3.5, p);
    canvas.drawCircle(const Offset(214, 360), 3.5, p);

    // Lower intake.
    canvas.drawPath(
      Path()
        ..moveTo(56, 364)
        ..quadraticBezierTo(100, 378, 160, 392)
        ..quadraticBezierTo(192, 402, 198, 424),
      p,
    );
    canvas.drawPath(
      Path()
        ..moveTo(56, 364)
        ..quadraticBezierTo(52, 410, 62, 452)
        ..quadraticBezierTo(170, 462, 290, 456),
      p,
    );

    // Splitter.
    canvas.drawPath(
      Path()
        ..moveTo(290, 410)
        ..quadraticBezierTo(210, 412, 188, 440)
        ..quadraticBezierTo(180, 460, 186, 470),
      p,
    );
    canvas.drawLine(const Offset(106, 470), const Offset(290, 470), p);

    // Tyre.
    canvas.drawPath(
      Path()
        ..moveTo(20, 440)
        ..lineTo(20, 496)
        ..quadraticBezierTo(22, 508, 40, 508)
        ..lineTo(92, 508)
        ..quadraticBezierTo(106, 508, 106, 494)
        ..lineTo(106, 470),
      p,
    );
    for (double x = 46; x <= 94; x += 9.5) {
      canvas.drawLine(Offset(x, 472), Offset(x, 506), p);
    }
  }

  @override
  bool shouldRepaint(_CarPainter old) =>
      old.color != color || old.headlightsOn != headlightsOn;
}
