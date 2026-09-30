import 'package:flutter/widgets.dart';

/// Design tokens taken from the IDR style sheet (Sometype Mono, 4-colour palette).
abstract final class IdrColors {
  /// Screen background — #171818.
  static const background = Color(0xFF171818);

  /// Tiles, cards, bottom sheets — #1e1f1f.
  static const surface = Color(0xFF1E1F1F);

  /// Slightly raised surface (pressed tiles, empty dot-matrix cells).
  static const surfaceHigh = Color(0xFF262727);

  /// Hairline borders around tiles and outlined dots.
  static const outline = Color(0xFF2E2F2F);

  /// Primary accent — #f06131 (toggles, route, DR trail, primary buttons).
  static const accent = Color(0xFFF06131);

  /// Positive / healthy — #48fa5d (GNSS dots, positive deltas).
  static const positive = Color(0xFF48FA5D);

  /// "You are here" dot while GPS is used — the familiar Google-Maps blue.
  static const location = Color(0xFF4285F4);

  /// Negative deltas use the accent tone, as in the mockups ("-5%").
  static const negative = Color(0xFFF06131);

  static const textPrimary = Color(0xFFF2F2F2);
  static const textSecondary = Color(0xFF8C8D8D);
  static const textMuted = Color(0xFF5C5D5D);

  /// Active tile ("bulb" tile in the mockup) is inverted: white body, dark glyph.
  static const tileActive = Color(0xFFFFFFFF);
  static const onTileActive = Color(0xFF171818);
}

abstract final class IdrRadius {
  static const tile = 22.0;
  static const card = 20.0;
  static const button = 18.0;
  static const sheet = 28.0;
}

abstract final class IdrSpace {
  static const xs = 4.0;
  static const sm = 8.0;
  static const md = 12.0;
  static const lg = 16.0;
  static const xl = 24.0;
  static const xxl = 32.0;

  /// Horizontal page gutter used on every screen.
  static const gutter = 28.0;
}
