import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../theme/theme.dart';

/// A road that runs through a tunnel. [progress] (0–1) is how far the car
/// has travelled. With [sensorNav] off the trail stops at the tunnel mouth
/// (GPS lost); with it on, an orange sensor trail carries through.
class TunnelRouteIllustration extends StatelessWidget {
  const TunnelRouteIllustration({
    super.key,
    required this.progress,
    required this.sensorNav,
  });

  final double progress;
  final bool sensorNav;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _TunnelRoutePainter(progress: progress, sensorNav: sensorNav),
      size: Size.infinite,
    );
  }
}

const _tunnelStart = 0.36;
const _tunnelEnd = 0.68;

class _TunnelRoutePainter extends CustomPainter {
  _TunnelRoutePainter({required this.progress, required this.sensorNav});

  final double progress;
  final bool sensorNav;

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width;
    final h = size.height;
    final road = Path()
      ..moveTo(w * 0.04, h * 0.78)
      ..cubicTo(w * 0.24, h * 0.78, w * 0.28, h * 0.5, w * 0.5, h * 0.5)
      ..cubicTo(w * 0.72, h * 0.5, w * 0.76, h * 0.22, w * 0.96, h * 0.22);
    final metric = road.computeMetrics().first;
    final len = metric.length;
    Path seg(double a, double b) =>
        metric.extractPath(len * a, len * math.max(a, b));

    // Road bed and tunnel.
    canvas.drawPath(
      road,
      Paint()
        ..color = IdrColors.outline
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );
    canvas.drawPath(
      seg(_tunnelStart, _tunnelEnd),
      Paint()
        ..color = IdrColors.surfaceHigh
        ..style = PaintingStyle.stroke
        ..strokeWidth = math.min(44, h * 0.16)
        ..strokeCap = StrokeCap.round,
    );
    final mid = metric
        .getTangentForOffset(len * (_tunnelStart + _tunnelEnd) / 2)!
        .position;
    final label = TextPainter(
      text: TextSpan(
        text: 'tunnel',
        style: IdrText.micro.copyWith(color: IdrColors.textMuted),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    label.paint(
      canvas,
      mid -
          Offset(
            label.width / 2,
            math.min(44, h * 0.16) / 2 + label.height + 6,
          ),
    );

    Paint trail(Color c) => Paint()
      ..color = c
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round;

    final p = progress.clamp(0.0, 1.0);
    canvas.drawPath(
      seg(0, math.min(p, _tunnelStart)),
      trail(IdrColors.textPrimary),
    );

    Offset head;
    var headColor = IdrColors.textPrimary;
    if (!sensorNav) {
      head = metric
          .getTangentForOffset(len * math.min(p, _tunnelStart))!
          .position;
      if (p > _tunnelStart) {
        // GPS lost: the trail freezes and a "?" pulses where it stopped.
        final q = TextPainter(
          text: TextSpan(
            text: '?',
            style: IdrText.title.copyWith(color: IdrColors.accent),
          ),
          textDirection: TextDirection.ltr,
        )..layout();
        q.paint(canvas, head + Offset(-q.width / 2, -q.height - 14));
        headColor = IdrColors.textMuted;
      }
    } else {
      if (p > _tunnelStart) {
        canvas.drawPath(
          seg(_tunnelStart, math.min(p, _tunnelEnd)),
          trail(IdrColors.accent),
        );
      }
      if (p > _tunnelEnd) {
        canvas.drawPath(seg(_tunnelEnd, p), trail(IdrColors.textPrimary));
      }
      head = metric.getTangentForOffset(len * p)!.position;
      if (p > _tunnelStart && p < _tunnelEnd) headColor = IdrColors.accent;
    }

    canvas.drawCircle(
      head,
      16,
      Paint()..color = headColor.withValues(alpha: 0.2),
    );
    canvas.drawCircle(head, 8, Paint()..color = headColor);
  }

  @override
  bool shouldRepaint(_TunnelRoutePainter old) =>
      old.progress != progress || old.sensorNav != sensorNav;
}

/// A phone held in a car mount, with motion "waves" coming off it.
class PhoneMountIllustration extends StatelessWidget {
  const PhoneMountIllustration({super.key, this.pulse = 0});

  /// 0–1, drives the wave animation.
  final double pulse;

  @override
  Widget build(BuildContext context) =>
      CustomPaint(painter: _PhoneMountPainter(pulse), size: Size.infinite);
}

class _PhoneMountPainter extends CustomPainter {
  _PhoneMountPainter(this.pulse);

  final double pulse;

  @override
  void paint(Canvas canvas, Size size) {
    final line = Paint()
      ..color = IdrColors.textPrimary
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.8
      ..strokeCap = StrokeCap.round;

    final ph = size.height * 0.62;
    final pw = ph * 0.5;
    final c = Offset(size.width * 0.42, size.height * 0.42);
    final phone = RRect.fromRectAndRadius(
      Rect.fromCenter(center: c, width: pw, height: ph),
      Radius.circular(pw * 0.18),
    );
    canvas.drawRRect(phone, line);
    canvas.drawRRect(
      phone.deflate(pw * 0.08),
      line..color = IdrColors.textMuted,
    );
    line.color = IdrColors.textPrimary;

    // Clamp arms, stem and suction base.
    final armY = c.dy;
    canvas.drawLine(
      Offset(phone.left - 10, armY - 22),
      Offset(phone.left - 10, armY + 22),
      line,
    );
    canvas.drawLine(
      Offset(phone.right + 10, armY - 22),
      Offset(phone.right + 10, armY + 22),
      line,
    );
    canvas.drawLine(
      Offset(phone.left - 10, armY),
      Offset(phone.left, armY),
      line,
    );
    canvas.drawLine(
      Offset(phone.right, armY),
      Offset(phone.right + 10, armY),
      line,
    );
    final stemTop = Offset(c.dx, phone.bottom + 6);
    final stemBottom = Offset(c.dx, size.height * 0.9);
    canvas.drawLine(stemTop, stemBottom, line);
    canvas.drawArc(
      Rect.fromCenter(
        center: stemBottom + const Offset(0, 8),
        width: pw * 0.9,
        height: 22,
      ),
      math.pi,
      math.pi,
      false,
      line,
    );

    // Orange dot on screen = the model running.
    canvas.drawCircle(c, 7, Paint()..color = IdrColors.accent);

    // Motion waves.
    for (var i = 0; i < 3; i++) {
      final t = (pulse + i / 3) % 1.0;
      final r = pw * 0.3 + t * pw * 0.9;
      canvas.drawArc(
        Rect.fromCircle(center: Offset(phone.right + 10, c.dy), radius: r),
        -math.pi / 4,
        math.pi / 2,
        false,
        Paint()
          ..color = IdrColors.accent.withValues(alpha: (1 - t) * 0.8)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2
          ..strokeCap = StrokeCap.round,
      );
    }
  }

  @override
  bool shouldRepaint(_PhoneMountPainter old) => old.pulse != pulse;
}

/// The IDR wordmark: "IDR" with an orange dot.
class IdrWordmark extends StatelessWidget {
  const IdrWordmark({super.key, this.size = 56});

  final double size;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        Text(
          'IDR',
          style: IdrText.display.copyWith(
            fontSize: size,
            height: 1,
            fontWeight: FontWeight.w500,
          ),
        ),
        Padding(
          padding: EdgeInsets.only(left: size * 0.08, bottom: size * 0.12),
          child: Container(
            width: size * 0.2,
            height: size * 0.2,
            decoration: const BoxDecoration(
              color: IdrColors.accent,
              shape: BoxShape.circle,
            ),
          ),
        ),
      ],
    );
  }
}
