import 'package:flutter/material.dart';

import '../theme/theme.dart';

/// Tall tile with a percentage and a vertical orange slider ("82%" in the mockup).
///
/// Read-only when [onChanged] is null.
class VerticalGauge extends StatelessWidget {
  const VerticalGauge({
    super.key,
    required this.value,
    this.onChanged,
    this.caption,
  });

  /// 0.0 – 1.0
  final double value;
  final ValueChanged<double>? onChanged;
  final String? caption;

  @override
  Widget build(BuildContext context) {
    final v = value.clamp(0.0, 1.0);
    return Container(
      decoration: BoxDecoration(
        color: IdrColors.surface,
        borderRadius: BorderRadius.circular(IdrRadius.tile),
      ),
      padding: const EdgeInsets.fromLTRB(
        IdrSpace.sm,
        30,
        IdrSpace.sm,
        IdrSpace.xl,
      ),
      child: Column(
        children: [
          Text(
            '${(v * 100).round()}%',
            style: IdrText.body.copyWith(fontSize: 18),
          ),
          if (caption != null) ...[
            const SizedBox(height: 2),
            Text(caption!, style: IdrText.micro, textAlign: TextAlign.center),
          ],
          const SizedBox(height: IdrSpace.xl),
          Expanded(
            child: LayoutBuilder(
              builder: (context, box) {
                void update(Offset local) {
                  final f = 1 - (local.dy / box.maxHeight);
                  onChanged?.call(f.clamp(0.0, 1.0));
                }

                return GestureDetector(
                  onVerticalDragUpdate: onChanged == null
                      ? null
                      : (d) => update(d.localPosition),
                  onTapDown: onChanged == null
                      ? null
                      : (d) => update(d.localPosition),
                  child: CustomPaint(
                    size: Size(box.maxWidth, box.maxHeight),
                    painter: _GaugePainter(v),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

class _GaugePainter extends CustomPainter {
  _GaugePainter(this.value);

  final double value;

  @override
  void paint(Canvas canvas, Size size) {
    const knobR = 19.0;
    final x = size.width / 2;
    final top = knobR;
    final bottom = size.height;
    final knobY = top + (bottom - top) * (1 - value);

    final track = Paint()
      ..color = IdrColors.textMuted
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;
    final fill = Paint()
      ..color = IdrColors.accent
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;

    canvas.drawLine(Offset(x, 0), Offset(x, knobY), track);
    canvas.drawLine(Offset(x, knobY), Offset(x, bottom), fill);
    canvas.drawCircle(
      Offset(x, knobY),
      knobR,
      Paint()..color = IdrColors.accent,
    );
  }

  @override
  bool shouldRepaint(_GaugePainter old) => old.value != value;
}
