import 'package:flutter/material.dart';

import '../theme/theme.dart';

class IdrNavItem {
  const IdrNavItem({
    required this.icon,
    required this.activeIcon,
    required this.label,
  });

  final IconData icon;
  final IconData activeIcon;
  final String label;
}

/// Bottom bar: selected item is white/filled, others muted/outlined.
class IdrBottomNav extends StatelessWidget {
  const IdrBottomNav({
    super.key,
    required this.items,
    required this.index,
    required this.onSelect,
  });

  final List<IdrNavItem> items;
  final int index;
  final ValueChanged<int> onSelect;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      top: false,
      child: SizedBox(
        height: 76,
        child: Row(
          children: [
            for (var i = 0; i < items.length; i++)
              Expanded(
                child: Semantics(
                  button: true,
                  selected: i == index,
                  label: items[i].label,
                  excludeSemantics: true,
                  child: GestureDetector(
                    behavior: HitTestBehavior.opaque,
                    onTap: () => onSelect(i),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          i == index ? items[i].activeIcon : items[i].icon,
                          size: 28,
                          color: i == index
                              ? IdrColors.textPrimary
                              : IdrColors.textMuted,
                        ),
                        const SizedBox(height: 4),
                        Text(
                          items[i].label,
                          style: IdrText.micro.copyWith(
                            fontSize: 11,
                            color: i == index
                                ? IdrColors.textPrimary
                                : IdrColors.textMuted,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
