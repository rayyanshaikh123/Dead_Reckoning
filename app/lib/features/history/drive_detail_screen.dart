import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:latlong2/latlong.dart';
import 'package:share_plus/share_plus.dart';

import '../../core/theme/theme.dart';
import '../../core/widgets/widgets.dart';
import '../../data/drives/drive_store.dart';
import '../../state/drive_recorder.dart';
import '../shell/page_scaffold.dart';
import 'history_screen.dart';

/// Report for one drive: route (GPS white, IDR estimate orange), totals,
/// every outage with its exit error, and the sensor log export.
class DriveDetailScreen extends ConsumerWidget {
  const DriveDetailScreen({super.key, required this.id});

  final String id;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final drive = ref
        .watch(drivesProvider)
        .value
        ?.where((d) => d.id == id)
        .firstOrNull;
    if (drive == null) {
      return const Scaffold(
        body: Center(child: Text('Drive not found', style: IdrText.small)),
      );
    }
    return Scaffold(
      body: IdrScrollPage(
        children: [
          IdrHeader(
            title: formatDriveTitle(drive.start),
            titleStyle: IdrText.title.copyWith(fontSize: 22),
            subtitle: drive.vehicleName,
            action: const CloseAction(),
          ),
          const SizedBox(height: IdrSpace.xl),
          if (drive.track.length > 1) _RouteMap(drive: drive),
          const SizedBox(height: IdrSpace.lg),
          TileGrid(
            children: [
              StatTile(
                label: 'Distance',
                value: '${(drive.distanceM / 1000).toStringAsFixed(1)} km',
              ),
              StatTile(label: 'Time', value: formatDuration(drive.duration)),
              StatTile(
                label: 'Avg speed',
                value: '${(drive.avgSpeedMs * 3.6).round()} km/h',
              ),
              StatTile(
                label: 'Top speed',
                value: '${(drive.maxSpeedMs * 3.6).round()} km/h',
              ),
            ],
          ),
          const SectionLabel('without gps'),
          if (drive.outages.isEmpty)
            Text(
              'GPS never dropped on this drive.',
              style: IdrText.label.copyWith(fontSize: 15),
            )
          else
            for (final (i, o) in drive.outages.indexed)
              _OutageRow(index: i + 1, outage: o),
          ..._mountComparison(drive.outages),
          if (drive.gyroRmseXKmh != null && drive.gyroRmseYKmh != null) ...[
            const SectionLabel('model check'),
            KeyValueRow(
              label: 'Gyro map X',
              value: '±${drive.gyroRmseXKmh!.toStringAsFixed(1)} km/h',
            ),
            KeyValueRow(
              label: 'Gyro map Y',
              value: '±${drive.gyroRmseYKmh!.toStringAsFixed(1)} km/h',
            ),
            Text(
              'Speed error against GPS for the two candidate sensor mappings. Lower is better.',
              style: IdrText.small,
            ),
          ],
          const SizedBox(height: IdrSpace.xl),
          if (drive.hasSensorLog) ...[
            LinkTile(
              label: 'Share sensor log',
              caption: 'CSV at 10 Hz, IO-VNBD style — for retraining',
              onTap: () => _share(context, ref, drive),
            ),
            const SizedBox(height: 8),
          ],
          PrimaryButton(
            'Delete drive',
            outlined: true,
            onPressed: () => _delete(context, ref, drive),
          ),
        ],
      ),
    );
  }

  Future<void> _share(
    BuildContext context,
    WidgetRef ref,
    DriveSummary d,
  ) async {
    // Where the share sheet anchors on iPad; read before any await.
    final box = context.findRenderObject() as RenderBox?;
    final f = await ref.read(driveStoreProvider).sensorLog(d.id);
    if (!await f.exists()) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No sensor log for this drive')),
        );
      }
      return;
    }
    await SharePlus.instance.share(
      ShareParams(
        files: [XFile(f.path, mimeType: 'text/csv')],
        subject: 'IDR drive ${d.id}',
        sharePositionOrigin: box == null
            ? null
            : box.localToGlobal(Offset.zero) & box.size,
      ),
    );
  }

  Future<void> _delete(
    BuildContext context,
    WidgetRef ref,
    DriveSummary d,
  ) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: IdrColors.surface,
        title: const Text('Delete this drive?', style: IdrText.body),
        content: Text(
          'Its route and sensor log are removed from this phone.',
          style: IdrText.small,
        ),
        actions: [
          BracketButton('cancel', onTap: () => Navigator.of(ctx).pop(false)),
          const SizedBox(width: IdrSpace.sm),
          BracketButton(
            'delete',
            style: IdrText.body.copyWith(color: IdrColors.accent),
            onTap: () => Navigator.of(ctx).pop(true),
          ),
        ],
      ),
    );
    if (ok != true) return;
    await ref.read(driveStoreProvider).delete(d.id);
    ref.invalidate(drivesProvider);
    if (context.mounted) Navigator.of(context).maybePop();
  }
}

/// Mounted vs in-hand exit error, once a drive has outages of both kinds
/// (e.g. the same tunnel test done both ways).
List<Widget> _mountComparison(List<Outage> outages) {
  double? avg(bool handheld) {
    final v = [
      for (final o in outages)
        if (o.handheld == handheld && o.exitErrorPct != null) o.exitErrorPct!,
    ];
    return v.isEmpty ? null : v.reduce((a, b) => a + b) / v.length;
  }

  final mounted = avg(false), inHand = avg(true);
  if (mounted == null || inHand == null) return const [];
  return [
    const SectionLabel('mounted vs in hand'),
    KeyValueRow(label: 'Mounted', value: '${mounted.toStringAsFixed(1)}% off'),
    KeyValueRow(label: 'In hand', value: '${inHand.toStringAsFixed(1)}% off'),
    Text(
      'Average exit error as a share of the distance driven without GPS.',
      style: IdrText.small,
    ),
  ];
}

class _RouteMap extends StatelessWidget {
  const _RouteMap({required this.drive});

  final DriveSummary drive;

  @override
  Widget build(BuildContext context) {
    // Consecutive points of the same kind form one polyline.
    final segments = <(bool, List<LatLng>)>[];
    for (final p in drive.track) {
      final ll = LatLng(p.lat, p.lon);
      if (segments.isEmpty || segments.last.$1 != p.gps) {
        segments.add((
          p.gps,
          [if (segments.isNotEmpty) segments.last.$2.last, ll],
        ));
      } else {
        segments.last.$2.add(ll);
      }
    }
    final all = [for (final p in drive.track) LatLng(p.lat, p.lon)];
    return ClipRRect(
      borderRadius: BorderRadius.circular(IdrRadius.tile),
      child: SizedBox(
        height: 260,
        child: IdrMap(
          interactive: false,
          initialCameraFit: CameraFit.coordinates(
            coordinates: all,
            padding: const EdgeInsets.all(28),
          ),
          layers: [
            PolylineLayer(
              polylines: [
                for (final (gps, pts) in segments)
                  Polyline(
                    points: pts,
                    color: gps ? IdrColors.textPrimary : IdrColors.accent,
                    strokeWidth: gps ? 3 : 5,
                  ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

class _OutageRow extends StatelessWidget {
  const _OutageRow({required this.index, required this.outage});

  final int index;
  final Outage outage;

  @override
  Widget build(BuildContext context) {
    final o = outage;
    final pct = o.exitErrorPct;
    final pass = o.passesSih;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: IdrSpace.sm),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '${o.simulated ? 'Tunnel test' : 'Outage'} $index · ${o.seconds.round()} s',
                  style: IdrText.body,
                ),
                const SizedBox(height: 2),
                Text(
                  '${o.distanceM.round()} m ${o.onRoad ? 'on the road map' : 'by heading'}'
                  '${o.handheld ? ' · phone in hand' : ''}',
                  style: IdrText.small,
                ),
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                o.exitErrorM == null
                    ? '--'
                    : 'off by ${o.exitErrorM!.round()} m',
                style: IdrText.body.copyWith(
                  color: pass == null
                      ? null
                      : (pass ? IdrColors.positive : IdrColors.negative),
                ),
              ),
              if (pct != null)
                Text(
                  '${pct.toStringAsFixed(1)}% of distance',
                  style: IdrText.small,
                ),
            ],
          ),
        ],
      ),
    );
  }
}
