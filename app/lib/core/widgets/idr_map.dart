import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';

import '../theme/theme.dart';

/// Map tiles. Defaults to the standard OpenStreetMap tiles (no API key),
/// darkened in-app by [_darken]. To use a keyed provider that already has a
/// dark style, build with e.g.
/// `--dart-define=MAP_TILES=https://…/{z}/{x}/{y}.png?api_key=…`
/// (the darkening filter is then skipped).
const _customTiles = String.fromEnvironment('MAP_TILES');
const _osmTiles = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';

/// Turns the light OSM style into the app's dark look: inverted luminance
/// scaled so land sits at the app background (#171818), roads and labels
/// become light-grey lines. Out = 0.45 · (255 − luminance) + 16, per channel.
const _darken = ColorFilter.matrix([
  -0.0957, -0.3218, -0.0325, 0, 130.75, //
  -0.0957, -0.3218, -0.0325, 0, 130.75, //
  -0.0957, -0.3218, -0.0325, 0, 130.75, //
  0, 0, 0, 1, 0, //
]);

/// Dark basemap (OpenStreetMap data) with IDR styling.
/// Pass overlay layers (polylines, markers) as [layers].
class IdrMap extends StatelessWidget {
  const IdrMap({
    super.key,
    required this.layers,
    this.controller,
    this.initialCenter = const LatLng(52.4068, -1.5197),
    this.initialZoom = 15,
    this.initialCameraFit,
    this.interactive = true,
    this.onMapReady,
    this.onUserGesture,
  });

  /// Set to false in widget tests: no tile downloads or tile cache.
  static bool showTiles = true;

  final List<Widget> layers;
  final MapController? controller;
  final LatLng initialCenter;
  final double initialZoom;
  final CameraFit? initialCameraFit;
  final bool interactive;
  final VoidCallback? onMapReady;

  /// Called when the user drags or zooms the map.
  final VoidCallback? onUserGesture;

  @override
  Widget build(BuildContext context) {
    return FlutterMap(
      mapController: controller,
      options: MapOptions(
        initialCenter: initialCenter,
        initialZoom: initialZoom,
        initialCameraFit: initialCameraFit,
        backgroundColor: const Color(0xFF151616),
        onMapReady: onMapReady,
        onPositionChanged: (_, hasGesture) {
          if (hasGesture) onUserGesture?.call();
        },
        minZoom: 3,
        maxZoom: 19,
        interactionOptions: InteractionOptions(
          flags: interactive
              ? InteractiveFlag.all & ~InteractiveFlag.rotate
              : InteractiveFlag.none,
        ),
      ),
      children: [
        if (showTiles)
          TileLayer(
            urlTemplate: _customTiles.isEmpty ? _osmTiles : _customTiles,
            // Identifies the app to the tile server, as the OSM tile usage
            // policy requires.
            userAgentPackageName: 'com.sihidr.idrApp',
            maxNativeZoom: 19,
            tileBuilder: _customTiles.isEmpty
                ? (context, tile, _) =>
                      ColorFiltered(colorFilter: _darken, child: tile)
                : null,
          ),
        ...layers,
        const _Attribution(),
      ],
    );
  }
}

class _Attribution extends StatelessWidget {
  const _Attribution();

  @override
  Widget build(BuildContext context) {
    return Align(
      alignment: Alignment.topRight,
      child: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(6),
          child: Text(
            '© OpenStreetMap contributors',
            style: IdrText.micro.copyWith(
              fontSize: 9,
              color: IdrColors.textMuted,
            ),
          ),
        ),
      ),
    );
  }
}

/// Round marker: filled dot with a soft halo.
class MapDot extends StatelessWidget {
  const MapDot({
    super.key,
    required this.color,
    this.hollow = false,
    this.size = 18,
  });

  final Color color;
  final bool hollow;
  final double size;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: hollow ? IdrColors.background : color,
        border: Border.all(color: color, width: hollow ? 3 : 0),
        boxShadow: [
          BoxShadow(
            color: color.withValues(alpha: 0.35),
            blurRadius: 12,
            spreadRadius: 4,
          ),
        ],
      ),
    );
  }
}
