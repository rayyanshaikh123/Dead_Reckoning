import 'package:flutter/material.dart';

import 'tokens.dart';
import 'typography.dart';

export 'tokens.dart';
export 'typography.dart';

ThemeData buildIdrTheme() {
  const scheme = ColorScheme.dark(
    surface: IdrColors.background,
    onSurface: IdrColors.textPrimary,
    primary: IdrColors.accent,
    onPrimary: IdrColors.textPrimary,
    secondary: IdrColors.positive,
    onSecondary: IdrColors.background,
    error: IdrColors.negative,
    outline: IdrColors.outline,
    surfaceContainer: IdrColors.surface,
    surfaceContainerHigh: IdrColors.surfaceHigh,
  );

  return ThemeData(
    useMaterial3: true,
    brightness: Brightness.dark,
    colorScheme: scheme,
    fontFamily: 'SometypeMono',
    scaffoldBackgroundColor: IdrColors.background,
    canvasColor: IdrColors.background,
    splashFactory: NoSplash.splashFactory,
    highlightColor: Colors.transparent,
    textTheme: const TextTheme(
      displayLarge: IdrText.display,
      titleLarge: IdrText.title,
      titleMedium: IdrText.subtitle,
      bodyLarge: IdrText.body,
      bodyMedium: IdrText.body,
      labelLarge: IdrText.label,
      bodySmall: IdrText.small,
      labelSmall: IdrText.micro,
    ),
    dividerTheme: const DividerThemeData(
      color: IdrColors.outline,
      thickness: 1,
      space: 1,
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: IdrColors.surface,
      hintStyle: IdrText.body.copyWith(color: IdrColors.textMuted),
      contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 18),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(IdrRadius.button),
        borderSide: const BorderSide(color: IdrColors.outline),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(IdrRadius.button),
        borderSide: const BorderSide(color: IdrColors.outline),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(IdrRadius.button),
        borderSide: const BorderSide(color: IdrColors.accent, width: 1.4),
      ),
    ),
    textSelectionTheme: const TextSelectionThemeData(
      cursorColor: IdrColors.accent,
      selectionHandleColor: IdrColors.accent,
    ),
    bottomSheetTheme: const BottomSheetThemeData(
      backgroundColor: IdrColors.surface,
      modalBackgroundColor: IdrColors.surface,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(
          top: Radius.circular(IdrRadius.sheet),
        ),
      ),
    ),
    snackBarTheme: const SnackBarThemeData(
      backgroundColor: IdrColors.surfaceHigh,
      contentTextStyle: IdrText.body,
      behavior: SnackBarBehavior.floating,
    ),
  );
}
