import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:share_plus/share_plus.dart';

import '../../app/routes.dart';
import '../../core/theme/theme.dart';
import '../../core/widgets/widgets.dart';
import '../../data/diagnostics/error_log.dart';
import '../../data/model_loader.dart';
import '../../data/roads/road_repository.dart';
import '../../data/sources/gnss_source.dart';
import '../../state/live_engine.dart';
import '../../state/settings.dart';
import '../shell/page_scaffold.dart';

/// Settings tab: car name, navigation behaviour, permissions, about.
class ProfileScreen extends ConsumerStatefulWidget {
  const ProfileScreen({super.key});

  @override
  ConsumerState<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends ConsumerState<ProfileScreen> {
  bool _modelDetails = false;

  Future<void> _rename(String current) async {
    final name = await showDialog<String>(
      context: context,
      builder: (_) => _RenameDialog(initial: current),
    );
    if (name != null) {
      await ref.read(appSettingsProvider.notifier).setVehicleName(name);
    }
  }

  @override
  Widget build(BuildContext context) {
    final settings = ref.watch(appSettingsProvider);
    final access = ref.watch(locationAccessProvider);
    final location = ref.read(locationAccessProvider.notifier);
    final modelStatus = ref.watch(speedModelProvider);
    final engine = ref.watch(liveEngineProvider);
    final roads = ref.watch(roadProvider);
    final roadStats = ref.watch(roadCacheStatsProvider).value;

    return IdrScrollPage(
      children: [
        IdrHeader(title: settings.vehicleName, subtitle: 'Settings'),
        const SectionLabel('your car'),
        _Row(
          title: 'Name',
          value: settings.vehicleName,
          action: BracketButton(
            'edit',
            onTap: () => _rename(settings.vehicleName),
          ),
        ),
        _Row(
          title: 'Phone mount',
          value: engine.aligned
              ? 'Learned from your driving'
              : 'Learning while you drive — ${(engine.alignmentProgress * 100).round()}%',
          valueColor: engine.aligned ? IdrColors.positive : null,
        ),
        _Row(
          title: 'Idle vibration',
          value:
              'Measured when parked · ${engine.idleBaseline.toStringAsFixed(4)}',
        ),
        const SectionLabel('navigation'),
        _Row(
          title: 'Auto sensor mode',
          value: 'Switch to sensors when GPS drops',
          action: IdrToggle(
            value: settings.autoSensorNav,
            onChanged: ref.read(appSettingsProvider.notifier).setAutoSensorNav,
          ),
        ),
        _Row(
          title: 'Run in background',
          value: 'Keeps working with the screen off or in another app',
          action: IdrToggle(
            value: settings.runInBackground,
            onChanged: ref
                .read(appSettingsProvider.notifier)
                .setRunInBackground,
          ),
        ),
        const SectionLabel('drives'),
        _Row(
          title: 'Record drives',
          value: 'Starts automatically when you drive',
          action: IdrToggle(
            value: settings.autoRecord,
            onChanged: ref.read(appSettingsProvider.notifier).setAutoRecord,
          ),
        ),
        _Row(
          title: 'Save sensor logs',
          value: 'About 7 MB per hour, kept on this phone',
          action: IdrToggle(
            value: settings.saveSensorLogs,
            onChanged: ref.read(appSettingsProvider.notifier).setSaveSensorLogs,
          ),
        ),
        const SectionLabel('permissions'),
        _Row(
          title: 'Location',
          value: switch (access) {
            LocationAccess.granted => 'Allowed',
            LocationAccess.serviceOff => 'Location services off',
            LocationAccess.deniedForever => 'Blocked',
            LocationAccess.unknown => 'Checking…',
            LocationAccess.denied => 'Not allowed yet',
          },
          valueColor: access == LocationAccess.granted
              ? IdrColors.positive
              : IdrColors.accent,
          action: switch (access) {
            LocationAccess.granted || LocationAccess.unknown => null,
            LocationAccess.denied => BracketButton(
              'allow',
              onTap: location.request,
            ),
            _ => BracketButton('open', onTap: location.openSettings),
          },
        ),
        const SectionLabel('offline'),
        _Row(
          title: 'Road map',
          value: [
            switch (roads.status) {
              RoadStatus.ready => 'Ready for this area',
              RoadStatus.downloading => 'Downloading nearby roads…',
              RoadStatus.offline when roads.data != null =>
                'No connection — using saved roads',
              RoadStatus.offline => 'No connection and no saved roads here',
              RoadStatus.idle => 'Downloads automatically while GPS is on',
            },
            if (roadStats != null && roadStats.areas > 0)
              '${roadStats.areas} area${roadStats.areas == 1 ? '' : 's'} saved '
                  '(${(roadStats.bytes / 1024).ceil()} KB)',
          ].join(' · '),
          valueColor: roads.status == RoadStatus.ready
              ? IdrColors.positive
              : null,
          action: (roadStats?.areas ?? 0) > 0
              ? BracketButton(
                  'clear',
                  onTap: ref.read(roadProvider.notifier).clearCache,
                )
              : null,
        ),
        const _Row(
          title: 'Map tiles',
          value: 'Areas you view are saved for offline use',
        ),
        const SectionLabel('about'),
        const _Row(title: 'IDR', value: 'Version 1.0 · runs fully offline'),
        _Row(
          title: 'AI model',
          value: modelStatus.when(
            data: (m) =>
                'IDNN v5 · ready · ${m.msPerStep.toStringAsFixed(2)} ms/step',
            loading: () => 'IDNN v5 · loading…',
            error: (_, _) => 'IDNN v5 · failed to load',
          ),
          valueColor: modelStatus.hasError
              ? IdrColors.accent
              : (modelStatus.hasValue ? IdrColors.positive : null),
          action: BracketButton(
            _modelDetails ? 'hide' : 'details',
            onTap: () => setState(() => _modelDetails = !_modelDetails),
          ),
        ),
        if (_modelDetails) ...[
          const SizedBox(height: IdrSpace.sm),
          const KeyValueRow(label: 'Input', value: '21 × 18 = 378'),
          const KeyValueRow(label: 'Hidden', value: '256·128·64'),
          const KeyValueRow(label: 'Params', value: '139 137'),
          const KeyValueRow(label: 'Tested on', value: '105 km blind'),
          const KeyValueRow(label: 'Tunnel drift', value: '2–10%'),
          KeyValueRow(
            label: 'Gyro check',
            value: engine.gyroCheck.samples < 600
                ? 'needs a drive'
                : 'X ±${engine.gyroCheck.rmseXKmh.toStringAsFixed(1)} · '
                      'Y ±${engine.gyroCheck.rmseYKmh.toStringAsFixed(1)} km/h',
          ),
        ],
        const SizedBox(height: IdrSpace.xl),
        LinkTile(
          label: 'Report a problem',
          caption: 'Share the error log from this phone',
          onTap: () => _shareErrorLog(context),
        ),
        const SizedBox(height: 8),
        LinkTile(
          label: 'Show intro again',
          onTap: () async {
            await ref.read(appSettingsProvider.notifier).resetOnboarding();
            if (context.mounted) context.go(Routes.onboarding);
          },
        ),
      ],
    );
  }
}

Future<void> _shareErrorLog(BuildContext context) async {
  // Where the share sheet anchors on iPad; read before any await.
  final box = context.findRenderObject() as RenderBox?;
  final origin = box == null ? null : box.localToGlobal(Offset.zero) & box.size;
  final messenger = ScaffoldMessenger.of(context);
  if (!await ErrorLog.hasEntries()) {
    messenger.showSnackBar(
      const SnackBar(content: Text('No errors recorded — all good')),
    );
    return;
  }
  final f = await ErrorLog.file();
  await SharePlus.instance.share(
    ShareParams(
      files: [XFile(f.path, mimeType: 'text/plain')],
      subject: 'IDR error log',
      sharePositionOrigin: origin,
    ),
  );
}

/// Settings row: title with a grey value underneath, optional control on the right.
class _Row extends StatelessWidget {
  const _Row({
    required this.title,
    required this.value,
    this.action,
    this.valueColor,
  });

  final String title;
  final String value;
  final Widget? action;
  final Color? valueColor;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: IdrSpace.sm),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(title, style: IdrText.body),
                const SizedBox(height: 2),
                Text(value, style: IdrText.small.copyWith(color: valueColor)),
              ],
            ),
          ),
          if (action != null) ...[const SizedBox(width: IdrSpace.md), action!],
        ],
      ),
    );
  }
}

/// Owns its text controller so it outlives the dialog's closing animation.
class _RenameDialog extends StatefulWidget {
  const _RenameDialog({required this.initial});

  final String initial;

  @override
  State<_RenameDialog> createState() => _RenameDialogState();
}

class _RenameDialogState extends State<_RenameDialog> {
  late final _ctl = TextEditingController(text: widget.initial);

  @override
  void dispose() {
    _ctl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      backgroundColor: IdrColors.surface,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(IdrRadius.card),
      ),
      title: const Text('Car name', style: IdrText.body),
      content: TextField(
        controller: _ctl,
        autofocus: true,
        style: IdrText.body,
        textCapitalization: TextCapitalization.words,
        onSubmitted: (v) => Navigator.of(context).pop(v),
      ),
      actions: [
        BracketButton('cancel', onTap: () => Navigator.of(context).pop()),
        const SizedBox(width: IdrSpace.sm),
        BracketButton(
          'save',
          style: IdrText.body.copyWith(color: IdrColors.accent),
          onTap: () => Navigator.of(context).pop(_ctl.text),
        ),
      ],
    );
  }
}
