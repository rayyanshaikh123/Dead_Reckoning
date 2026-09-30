import 'package:flutter/material.dart';

import '../../core/theme/theme.dart';
import '../../core/widgets/widgets.dart';
import 'menu_sheet.dart';

/// Standard page header: big title, grey subtitle, bracket action on the right.
class IdrHeader extends StatelessWidget {
  const IdrHeader({
    super.key,
    required this.title,
    this.subtitle,
    this.action,
    this.titleStyle,
  });

  final String title;
  final String? subtitle;
  final Widget? action;
  final TextStyle? titleStyle;

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: titleStyle ?? IdrText.title),
              if (subtitle != null) ...[
                const SizedBox(height: IdrSpace.sm),
                Text(subtitle!, style: IdrText.subtitle),
              ],
            ],
          ),
        ),
        Padding(
          padding: const EdgeInsets.only(top: 6),
          child:
              action ??
              BracketButton('menu', onTap: () => showIdrMenu(context)),
        ),
      ],
    );
  }
}

/// Scrollable page with the standard gutter, used by tabs and pushed routes.
class IdrScrollPage extends StatelessWidget {
  const IdrScrollPage({super.key, required this.children});

  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      bottom: false,
      child: ListView(
        padding: const EdgeInsets.fromLTRB(
          IdrSpace.gutter,
          IdrSpace.xl,
          IdrSpace.gutter,
          IdrSpace.xxl,
        ),
        children: children,
      ),
    );
  }
}

/// Header action for pushed routes.
class CloseAction extends StatelessWidget {
  const CloseAction({super.key});

  @override
  Widget build(BuildContext context) {
    return BracketButton(
      'close',
      onTap: () => Navigator.of(context).maybePop(),
    );
  }
}
