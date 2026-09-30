import 'package:flutter/widgets.dart';

import 'tokens.dart';

const _family = 'SometypeMono';

/// Type scale. Everything in the app is set in Sometype Mono.
abstract final class IdrText {
  /// Live speed on the home screen.
  static const hero = TextStyle(
    fontFamily: _family,
    fontSize: 64,
    height: 1.0,
    fontWeight: FontWeight.w400,
    color: IdrColors.textPrimary,
  );

  /// "≈125 kWh" style hero numbers.
  static const display = TextStyle(
    fontFamily: _family,
    fontSize: 40,
    height: 1.1,
    fontWeight: FontWeight.w400,
    color: IdrColors.textPrimary,
  );

  /// Screen titles — "BMW i7 (G70)".
  static const title = TextStyle(
    fontFamily: _family,
    fontSize: 28,
    height: 1.2,
    fontWeight: FontWeight.w400,
    color: IdrColors.textPrimary,
  );

  /// Screen subtitles — "Parked".
  static const subtitle = TextStyle(
    fontFamily: _family,
    fontSize: 22,
    height: 1.2,
    fontWeight: FontWeight.w400,
    color: IdrColors.textSecondary,
  );

  /// Values in stat tiles — "$56", "125 kWh".
  static const stat = TextStyle(
    fontFamily: _family,
    fontSize: 26,
    height: 1.2,
    fontWeight: FontWeight.w400,
    color: IdrColors.textPrimary,
  );

  static const body = TextStyle(
    fontFamily: _family,
    fontSize: 17,
    height: 1.35,
    fontWeight: FontWeight.w400,
    color: IdrColors.textPrimary,
  );

  /// Left-hand labels in key/value rows — "Mileage".
  static const label = TextStyle(
    fontFamily: _family,
    fontSize: 17,
    height: 1.35,
    fontWeight: FontWeight.w400,
    color: IdrColors.textSecondary,
  );

  static const small = TextStyle(
    fontFamily: _family,
    fontSize: 13,
    height: 1.3,
    fontWeight: FontWeight.w400,
    color: IdrColors.textSecondary,
  );

  /// Tiny chips and deltas — "+1%".
  static const micro = TextStyle(
    fontFamily: _family,
    fontSize: 12,
    height: 1.2,
    fontWeight: FontWeight.w500,
    color: IdrColors.textSecondary,
  );
}
