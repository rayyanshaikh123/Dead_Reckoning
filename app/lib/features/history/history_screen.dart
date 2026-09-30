import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../core/theme/theme.dart';
import '../../core/widgets/widgets.dart';
import '../../data/drives/drive_store.dart';
import '../../state/drive_recorder.dart';
import '../shell/page_scaffold.dart';

/// Past drives: the last 7 days at a glance, then every drive.
class HistoryScreen extends ConsumerWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final drives = ref.watch(drivesProvider);
    final recorder = ref.watch(driveRecorderProvider);

    return Scaffold(
      body: drives.when(
        loading: () => const Center(
          child: CircularProgressIndicator(color: IdrColors.accent),
        ),
        error: (e, _) => Center(child: Text('$e', style: IdrText.small)),
        data: (list) => IdrScrollPage(
          children: [
            ..._weekHeader(list),
            if (recorder.current != null) ...[
              const SizedBox(height: IdrSpace.xl),
              _RecordingTile(drive: recorder.current!),
            ],
            if (list.isEmpty && recorder.current == null)
              ..._empty()
            else ...[
              const SectionLabel('drives'),
              for (final d in list) ...[
                LinkTile(
                  label: formatDriveTitle(d.start),
                  caption: _caption(d),
                  onTap: () => context.push('/history/${d.id}'),
                ),
                const SizedBox(height: 8),
              ],
            ],
          ],
        ),
      ),
    );
  }

  List<Widget> _weekHeader(List<DriveSummary> list) {
    final today = DateUtils.dateOnly(DateTime.now());
    final days = [
      for (var i = 6; i >= 0; i--) today.subtract(Duration(days: i)),
    ];
    final km = [
      for (final day in days)
        list
            .where((d) => DateUtils.isSameDay(d.start, day))
            .fold<double>(0, (s, d) => s + d.distanceM / 1000),
    ];
    final week = list.where((d) => !d.start.isBefore(days.first)).toList();
    final total = km.fold<double>(0, (s, v) => s + v);
    final top = math.max(1.0, km.reduce(math.max));
    final outages = week.expand((d) => d.outages).toList();
    final exits = [
      for (final o in outages)
        if (o.exitErrorPct != null) o.exitErrorPct!,
    ];

    return [
      IdrHeader(
        title: '≈${total < 10 ? total.toStringAsFixed(1) : total.round()} km',
        titleStyle: IdrText.display,
        action: const CloseAction(),
      ),
      const SizedBox(height: IdrSpace.sm),
      const Text('[last 7 days]', style: IdrText.body),
      const SizedBox(height: IdrSpace.xl),
      DotMatrixChart(
        values: [for (final v in km) v / top],
        labels: [for (final d in days) 'MTWTFSS'[d.weekday - 1]],
        highlight: 6,
      ),
      const SizedBox(height: IdrSpace.xl),
      TileGrid(
        children: [
          StatTile(label: 'Drives', value: '${week.length}'),
          StatTile(
            label: 'Without GPS',
            value:
                '${(week.fold<double>(0, (s, d) => s + d.outageDistanceM) / 1000).toStringAsFixed(1)} km',
          ),
          StatTile(label: 'Outages', value: '${outages.length}'),
          StatTile(
            label: 'Exit error',
            value: exits.isEmpty
                ? '--'
                : '${(exits.reduce((a, b) => a + b) / exits.length).toStringAsFixed(1)}%',
          ),
        ],
      ),
    ];
  }

  List<Widget> _empty() => [
    const SizedBox(height: IdrSpace.xxl),
    const Text('No drives yet.', style: IdrText.title),
    const SizedBox(height: IdrSpace.md),
    Text(
      'Drives are recorded automatically once you’re moving. Each one shows '
      'its distance, any stretch without GPS, and how close IDR’s estimate '
      'was when GPS came back.',
      style: IdrText.label.copyWith(height: 1.5),
    ),
  ];

  String _caption(DriveSummary d) {
    final parts = [
      '${(d.distanceM / 1000).toStringAsFixed(1)} km',
      formatDuration(d.duration),
      if (d.outages.isNotEmpty)
        '${d.outages.length} outage${d.outages.length == 1 ? '' : 's'}',
      if (d.inProgress) 'interrupted',
    ];
    return parts.join(' · ');
  }
}

class _RecordingTile extends ConsumerWidget {
  const _RecordingTile({required this.drive});

  final DriveSummary drive;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return IdrTile(
      padding: const EdgeInsets.fromLTRB(
        IdrSpace.xl,
        IdrSpace.lg,
        IdrSpace.lg,
        IdrSpace.lg,
      ),
      child: Row(
        children: [
          Container(
            width: 10,
            height: 10,
            margin: const EdgeInsets.only(right: IdrSpace.md),
            decoration: const BoxDecoration(
              color: IdrColors.accent,
              shape: BoxShape.circle,
            ),
          ),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Recording this drive', style: IdrText.body),
                const SizedBox(height: 2),
                Text(
                  '${(drive.distanceM / 1000).toStringAsFixed(1)} km · ${formatDuration(drive.duration)}',
                  style: IdrText.small,
                ),
              ],
            ),
          ),
          BracketButton(
            'end',
            style: IdrText.body.copyWith(color: IdrColors.accent),
            onTap: ref.read(driveRecorderProvider.notifier).finish,
          ),
        ],
      ),
    );
  }
}

/// "Mon 29 Sep · 18:42"
String formatDriveTitle(DateTime t) {
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const months = [
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
    'Jul',
    'Aug',
    'Sep',
    'Oct',
    'Nov',
    'Dec',
  ];
  String two(int v) => v.toString().padLeft(2, '0');
  return '${days[t.weekday - 1]} ${t.day} ${months[t.month - 1]} · ${two(t.hour)}:${two(t.minute)}';
}

/// "1 h 05 min", "23 min", "40 s"
String formatDuration(Duration d) {
  if (d.inMinutes == 0) return '${d.inSeconds} s';
  if (d.inHours == 0) return '${d.inMinutes} min';
  return '${d.inHours} h ${(d.inMinutes % 60).toString().padLeft(2, '0')} min';
}
