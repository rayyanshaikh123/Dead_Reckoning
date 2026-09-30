import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../theme/theme.dart';

/// Column chart made of stacked dots with an optional overlay line
/// (the "≈125 kWh" chart). Each column fills `round(value * rows)` dots
/// from the bottom; [values] and [overlay] are normalised 0..1.
class DotMatrixChart extends StatelessWidget {
  const DotMatrixChart({
    super.key,
    required this.values,
    this.overlay,
    this.labels,
    this.rows = 5,
    this.highlight,
    this.color = IdrColors.accent,
  });

  final List<double> values;
  final List<double>? overlay;
  final List<String>? labels;
  final int rows;

  /// Index of a column to draw at full opacity while others are dimmed.
  final int? highlight;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, box) {
        final cols = values.length;
        final cell = box.maxWidth / cols;
        final labelRows = labels == null ? 0 : 1;
        final height =
            cell * (rows + labelRows) + (labelRows > 0 ? cell * 0.1 : 0);
        return SizedBox(
          width: box.maxWidth,
          height: height,
          child: CustomPaint(
            painter: _DotMatrixPainter(
              values: values,
              overlay: overlay,
              labels: labels,
              rows: rows,
              cell: cell,
              highlight: highlight,
              color: color,
            ),
          ),
        );
      },
    );
  }
}

class _DotMatrixPainter extends CustomPainter {
  _DotMatrixPainter({
    required this.values,
    required this.overlay,
    required this.labels,
    required this.rows,
    required this.cell,
    required this.highlight,
    required this.color,
  });

  final List<double> values;
  final List<double>? overlay;
  final List<String>? labels;
  final int rows;
  final double cell;
  final int? highlight;
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final r = cell * 0.46;
    final empty = Paint()..color = IdrColors.surfaceHigh;

    for (var c = 0; c < values.length; c++) {
      final filled = (values[c].clamp(0.0, 1.0) * rows).round();
      final dim = highlight != null && highlight != c;
      final fill = Paint()..color = dim ? color.withValues(alpha: 0.35) : color;
      final cx = cell * c + cell / 2;
      for (var row = 0; row < rows; row++) {
        final cy = cell * (rows - 1 - row) + cell / 2;
        canvas.drawCircle(Offset(cx, cy), r, row < filled ? fill : empty);
      }
    }

    final line = overlay;
    if (line != null && line.length == values.length) {
      final path = Path();
      for (var c = 0; c < line.length; c++) {
        final x = cell * c + cell / 2;
        final y = cell * rows * (1 - line[c].clamp(0.0, 1.0)) + cell * 0.1;
        if (c == 0) {
          path.moveTo(x, y);
        } else {
          final px = cell * (c - 1) + cell / 2;
          final py =
              cell * rows * (1 - line[c - 1].clamp(0.0, 1.0)) + cell * 0.1;
          final mx = (px + x) / 2;
          path.cubicTo(mx, py, mx, y, x, y);
        }
      }
      canvas.drawPath(
        path,
        Paint()
          ..color = const Color(0xFFE8C9BC).withValues(alpha: 0.55)
          ..style = PaintingStyle.stroke
          ..strokeWidth = math.max(2.0, cell * 0.06)
          ..strokeJoin = StrokeJoin.round
          ..strokeCap = StrokeCap.round,
      );
    }

    final l = labels;
    if (l != null) {
      final border = Paint()
        ..color = IdrColors.outline
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.2;
      final cy = cell * rows + cell * 0.1 + cell / 2;
      for (var c = 0; c < l.length && c < values.length; c++) {
        final cx = cell * c + cell / 2;
        canvas.drawCircle(Offset(cx, cy), r, border);
        final tp = TextPainter(
          text: TextSpan(
            text: l[c],
            style: IdrText.body.copyWith(fontSize: math.min(17, cell * 0.28)),
          ),
          textDirection: TextDirection.ltr,
        )..layout();
        tp.paint(canvas, Offset(cx - tp.width / 2, cy - tp.height / 2));
      }
    }
  }

  @override
  bool shouldRepaint(_DotMatrixPainter old) =>
      old.values != values ||
      old.overlay != overlay ||
      old.labels != labels ||
      old.highlight != highlight;
}
