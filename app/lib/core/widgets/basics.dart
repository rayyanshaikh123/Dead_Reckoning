import 'package:flutter/material.dart';

import '../theme/theme.dart';

/// Monospace text action in square brackets — "[menu]", "[april]".
class BracketButton extends StatelessWidget {
  const BracketButton(this.label, {super.key, this.onTap, this.style});

  final String label;
  final VoidCallback? onTap;
  final TextStyle? style;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      behavior: HitTestBehavior.opaque,
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: IdrSpace.xs),
        child: Text('[$label]', style: style ?? IdrText.body),
      ),
    );
  }
}

/// Rounded surface used for every tile and card.
class IdrTile extends StatelessWidget {
  const IdrTile({
    super.key,
    required this.child,
    this.active = false,
    this.onTap,
    this.padding = const EdgeInsets.all(IdrSpace.lg),
    this.radius = IdrRadius.tile,
    this.color,
  });

  final Widget child;
  final bool active;
  final VoidCallback? onTap;
  final EdgeInsetsGeometry padding;
  final double radius;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      behavior: HitTestBehavior.opaque,
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        curve: Curves.easeOut,
        padding: padding,
        decoration: BoxDecoration(
          color: active ? IdrColors.tileActive : (color ?? IdrColors.surface),
          borderRadius: BorderRadius.circular(radius),
          border: Border.all(
            color: active ? IdrColors.tileActive : IdrColors.outline,
          ),
        ),
        child: child,
      ),
    );
  }
}

/// Square tile holding a single glyph; inverts to white when [active].
class IconTile extends StatelessWidget {
  const IconTile({
    super.key,
    required this.icon,
    this.active = false,
    this.onTap,
    this.semanticLabel,
    this.label,
  });

  final IconData icon;
  final bool active;
  final VoidCallback? onTap;
  final String? semanticLabel;

  /// Optional caption under the glyph.
  final String? label;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: onTap != null,
      toggled: onTap != null ? active : null,
      label: semanticLabel ?? label,
      excludeSemantics: true,
      child: AspectRatio(
        aspectRatio: 1,
        child: IdrTile(
          active: active,
          onTap: onTap,
          padding: EdgeInsets.zero,
          child: Center(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  icon,
                  size: 28,
                  color: active
                      ? IdrColors.onTileActive
                      : IdrColors.textPrimary,
                ),
                if (label != null) ...[
                  const SizedBox(height: 6),
                  Text(
                    label!,
                    textAlign: TextAlign.center,
                    style: IdrText.micro.copyWith(
                      color: active
                          ? IdrColors.onTileActive
                          : IdrColors.textSecondary,
                    ),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}

/// Orange pill switch from the "Safe mode" row.
class IdrToggle extends StatelessWidget {
  const IdrToggle({super.key, required this.value, this.onChanged});

  final bool value;
  final ValueChanged<bool>? onChanged;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      toggled: value,
      child: GestureDetector(
        onTap: onChanged == null ? null : () => onChanged!(!value),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          curve: Curves.easeOut,
          width: 60,
          height: 36,
          padding: const EdgeInsets.all(4),
          decoration: BoxDecoration(
            color: value ? IdrColors.accent : IdrColors.surfaceHigh,
            borderRadius: BorderRadius.circular(18),
          ),
          child: AnimatedAlign(
            duration: const Duration(milliseconds: 200),
            curve: Curves.easeOut,
            alignment: value ? Alignment.centerRight : Alignment.centerLeft,
            child: Container(
              width: 28,
              height: 28,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: value ? IdrColors.textPrimary : IdrColors.textMuted,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

/// Row of filled/outlined dots — the battery indicator in the mockup.
class DotIndicator extends StatelessWidget {
  const DotIndicator({
    super.key,
    required this.filled,
    this.total = 5,
    this.color = IdrColors.positive,
    this.size = 18,
  });

  final int filled;
  final int total;
  final Color color;
  final double size;

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        for (var i = 0; i < total; i++)
          Container(
            width: size,
            height: size,
            margin: EdgeInsets.only(left: i == 0 ? 0 : size * 0.3),
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: i < filled ? color : Colors.transparent,
              border: i < filled
                  ? null
                  : Border.all(color: IdrColors.textMuted, width: 1.2),
            ),
          ),
      ],
    );
  }
}

/// Label/value row used on the dashboard ("Mileage   17 620 km").
class KeyValueRow extends StatelessWidget {
  const KeyValueRow({
    super.key,
    required this.label,
    required this.value,
    this.trailing,
    this.valueColor,
  });

  final String label;
  final String value;
  final Widget? trailing;
  final Color? valueColor;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 7),
      child: Row(
        children: [
          Expanded(flex: 10, child: Text(label, style: IdrText.label)),
          Expanded(
            flex: 9,
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    value,
                    style: IdrText.body.copyWith(color: valueColor),
                  ),
                ),
                ?trailing,
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// Stat card from the analytics screen ("Total charged / 125 kWh +1%").
class StatTile extends StatelessWidget {
  const StatTile({
    super.key,
    required this.label,
    required this.value,
    this.delta,
    this.deltaPositive = true,
  });

  final String label;
  final String value;
  final String? delta;
  final bool deltaPositive;

  @override
  Widget build(BuildContext context) {
    return IdrTile(
      padding: const EdgeInsets.fromLTRB(IdrSpace.xl, 20, IdrSpace.xl, 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: IdrText.label.copyWith(fontSize: 15),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
          const SizedBox(height: IdrSpace.sm),
          Row(
            crossAxisAlignment: CrossAxisAlignment.baseline,
            textBaseline: TextBaseline.alphabetic,
            children: [
              Flexible(
                child: FittedBox(
                  fit: BoxFit.scaleDown,
                  alignment: Alignment.centerLeft,
                  child: Text(value, style: IdrText.stat),
                ),
              ),
              if (delta != null) ...[
                const Spacer(),
                Text(
                  delta!,
                  style: IdrText.micro.copyWith(
                    fontSize: 14,
                    color: deltaPositive
                        ? IdrColors.positive
                        : IdrColors.negative,
                  ),
                ),
              ],
            ],
          ),
        ],
      ),
    );
  }
}

/// Full-width orange action — "Book charger".
class PrimaryButton extends StatelessWidget {
  const PrimaryButton(
    this.label, {
    super.key,
    this.onPressed,
    this.outlined = false,
  });

  final String label;
  final VoidCallback? onPressed;
  final bool outlined;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      child: GestureDetector(
        onTap: onPressed,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          height: 60,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: outlined ? Colors.transparent : IdrColors.accent,
            borderRadius: BorderRadius.circular(IdrRadius.button),
            border: Border.all(color: IdrColors.accent, width: 1.4),
          ),
          child: Text(
            label,
            style: IdrText.body.copyWith(
              color: outlined ? IdrColors.accent : IdrColors.textPrimary,
            ),
          ),
        ),
      ),
    );
  }
}

/// Wide tile with a label and a trailing arrow — "Report →".
class LinkTile extends StatelessWidget {
  const LinkTile({super.key, required this.label, this.onTap, this.caption});

  final String label;
  final String? caption;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    return IdrTile(
      onTap: onTap,
      padding: const EdgeInsets.symmetric(
        horizontal: IdrSpace.xl,
        vertical: 26,
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label, style: IdrText.body),
                if (caption != null) ...[
                  const SizedBox(height: 2),
                  Text(caption!, style: IdrText.small),
                ],
              ],
            ),
          ),
          const Icon(
            Icons.arrow_forward,
            size: 22,
            color: IdrColors.textPrimary,
          ),
        ],
      ),
    );
  }
}

/// Outlined round icon with a caption below — the "220 kW / 34 min" blocks.
class RoundInfo extends StatelessWidget {
  const RoundInfo({
    super.key,
    required this.icon,
    required this.line1,
    required this.line2,
  });

  final IconData icon;
  final String line1;
  final String line2;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: 56,
          height: 56,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            border: Border.all(color: IdrColors.outline, width: 1.2),
          ),
          child: Icon(icon, size: 24, color: IdrColors.textPrimary),
        ),
        const SizedBox(height: IdrSpace.xl),
        Text(line1, style: IdrText.body.copyWith(fontSize: 15)),
        Text(line2, style: IdrText.body.copyWith(fontSize: 15)),
      ],
    );
  }
}

/// Small status chip — used for pipeline stages (GATE / EKF / ZUPT ...).
class StageChip extends StatelessWidget {
  const StageChip(this.label, {super.key, this.active = false});

  final String label;
  final bool active;

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 180),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
      decoration: BoxDecoration(
        color: active ? IdrColors.accent : Colors.transparent,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: active ? IdrColors.accent : IdrColors.outline,
          width: 1.2,
        ),
      ),
      child: Text(
        label,
        style: IdrText.micro.copyWith(
          color: active ? IdrColors.textPrimary : IdrColors.textSecondary,
        ),
      ),
    );
  }
}

/// Section heading used on long screens.
class SectionLabel extends StatelessWidget {
  const SectionLabel(this.text, {super.key});

  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(top: IdrSpace.xl, bottom: IdrSpace.md),
      child: Text(
        text.toLowerCase(),
        style: IdrText.small.copyWith(color: IdrColors.textMuted),
      ),
    );
  }
}

/// Two-column tile grid with the 8 px gaps used in the mockups.
class TileGrid extends StatelessWidget {
  const TileGrid({super.key, required this.children});

  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        for (var i = 0; i < children.length; i += 2)
          Padding(
            padding: EdgeInsets.only(top: i == 0 ? 0 : 8),
            child: Row(
              children: [
                Expanded(child: children[i]),
                const SizedBox(width: 8),
                Expanded(
                  child: i + 1 < children.length
                      ? children[i + 1]
                      : const SizedBox(),
                ),
              ],
            ),
          ),
      ],
    );
  }
}
