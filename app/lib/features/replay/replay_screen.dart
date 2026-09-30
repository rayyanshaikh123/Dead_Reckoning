import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../core/theme/theme.dart';
import '../../core/widgets/widgets.dart';
import '../../data/benchmark.dart';
import '../shell/page_scaffold.dart';

/// Replay library: the five README tunnels from IO-VNBD Drive M. Each card
/// shows the benchmark result; tapping plays it through the on-device engine.
class ReplayScreen extends StatelessWidget {
  const ReplayScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IdrScrollPage(
        children: [
          const IdrHeader(
            title: 'Replay',
            subtitle: 'Real tunnel tests',
            action: CloseAction(),
          ),
          const SizedBox(height: IdrSpace.lg),
          Text(
            'Real tunnels from a 105 km drive the model never saw in training. '
            'Tap one to watch IDR run it live on this phone.',
            style: IdrText.label.copyWith(fontSize: 15, height: 1.5),
          ),
          const SizedBox(height: IdrSpace.xl),
          for (final sc in benchmarkScenarios) ...[
            _ScenarioCard(
              scenario: sc,
              onTap: () => context.push('/replay/${sc.id}'),
            ),
            const SizedBox(height: 8),
          ],
        ],
      ),
    );
  }
}

class _ScenarioCard extends StatelessWidget {
  const _ScenarioCard({required this.scenario, this.onTap});

  final BenchmarkScenario scenario;
  final VoidCallback? onTap;

  @override
  Widget build(BuildContext context) {
    final sc = scenario;
    return IdrTile(
      onTap: onTap,
      padding: const EdgeInsets.fromLTRB(IdrSpace.xl, 20, IdrSpace.xl, 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(child: Text(sc.name, style: IdrText.body)),
              Text(
                '[${sc.distanceM.round()}m]',
                style: IdrText.body.copyWith(fontSize: 15),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text('${sc.durationS} s outage', style: IdrText.small),
          const SizedBox(height: IdrSpace.lg),
          Row(
            children: [
              Expanded(
                child: _Metric(
                  label: 'IDR drift',
                  value: '${sc.v5AlongTrackPct}%',
                  good: sc.passesSih,
                ),
              ),
              Expanded(
                child: _Metric(
                  label: 'exit error',
                  value: '${sc.v5ExitErrorM.round()} m',
                ),
              ),
              Expanded(
                child: _Metric(
                  label: 'without AI',
                  value: '${sc.insDriftPct.round()}%',
                  good: false,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _Metric extends StatelessWidget {
  const _Metric({required this.label, required this.value, this.good});

  final String label;
  final String value;
  final bool? good;

  @override
  Widget build(BuildContext context) {
    final color = switch (good) {
      true => IdrColors.positive,
      false => IdrColors.negative,
      null => IdrColors.textPrimary,
    };
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: IdrText.micro),
        const SizedBox(height: 2),
        Text(value, style: IdrText.body.copyWith(color: color)),
      ],
    );
  }
}
