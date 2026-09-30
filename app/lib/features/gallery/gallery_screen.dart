import 'package:flutter/material.dart';

import '../../core/theme/theme.dart';
import '../../core/widgets/widgets.dart';
import '../shell/page_scaffold.dart';

/// Debug reference of every design-system component.
class GalleryScreen extends StatefulWidget {
  const GalleryScreen({super.key});

  @override
  State<GalleryScreen> createState() => _GalleryScreenState();
}

class _GalleryScreenState extends State<GalleryScreen> {
  bool _toggle = true;
  bool _tile = true;
  double _gauge = 0.82;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IdrScrollPage(
        children: [
          const IdrHeader(
            title: 'Gallery',
            subtitle: 'IDR design system',
            action: CloseAction(),
          ),
          const SectionLabel('colours'),
          const Row(
            children: [
              _Swatch(IdrColors.background, '#171818'),
              _Swatch(IdrColors.surface, '#1e1f1f'),
              _Swatch(IdrColors.accent, '#f06131'),
              _Swatch(IdrColors.positive, '#48fa5d'),
            ],
          ),
          const SectionLabel('type'),
          const Text('≈125 kWh', style: IdrText.display),
          const Text('Sometype Mono', style: IdrText.title),
          const Text('Parked', style: IdrText.subtitle),
          const Text('0 1 2 3 4 5 6 7 8 9', style: IdrText.body),
          const Text('Mileage', style: IdrText.label),
          const SectionLabel('controls'),
          Row(
            children: [
              IdrToggle(
                value: _toggle,
                onChanged: (v) => setState(() => _toggle = v),
              ),
              const SizedBox(width: IdrSpace.xl),
              const DotIndicator(filled: 3),
              const Spacer(),
              const BracketButton('menu'),
            ],
          ),
          const SizedBox(height: IdrSpace.lg),
          Wrap(
            spacing: IdrSpace.sm,
            runSpacing: IdrSpace.sm,
            children: const [
              StageChip('EKF', active: true),
              StageChip('ZUPT'),
              StageChip('CALIB'),
            ],
          ),
          const SectionLabel('tiles'),
          SizedBox(
            height: 300,
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                SizedBox(
                  width: 96,
                  child: VerticalGauge(
                    value: _gauge,
                    onChanged: (v) => setState(() => _gauge = v),
                  ),
                ),
                const SizedBox(width: 8),
                SizedBox(
                  width: 96,
                  child: Column(
                    children: [
                      IconTile(
                        icon: Icons.lightbulb_outline,
                        active: _tile,
                        onTap: () => setState(() => _tile = !_tile),
                      ),
                      const SizedBox(height: 8),
                      const IconTile(icon: Icons.lock_outline),
                    ],
                  ),
                ),
                const SizedBox(width: IdrSpace.lg),
                const Expanded(child: CarLineArt()),
              ],
            ),
          ),
          const SizedBox(height: 8),
          const TileGrid(
            children: [
              StatTile(label: 'Total charged', value: '125 kWh', delta: '+1%'),
              StatTile(
                label: 'Gas savings',
                value: '\$22',
                delta: '-5%',
                deltaPositive: false,
              ),
            ],
          ),
          const SizedBox(height: 8),
          const LinkTile(label: 'Report'),
          const SectionLabel('chart'),
          const DotMatrixChart(
            values: [0.6, 0.8, 0.2, 1.0, 0.4, 0.8, 0.4],
            overlay: [0.5, 0.75, 0.15, 0.95, 0.35, 0.72, 0.4],
            labels: ['15', '16', '17', '18', '19', '20', '21'],
          ),
          const SectionLabel('buttons'),
          const PrimaryButton('Book charger'),
          const SizedBox(height: 8),
          const PrimaryButton('Outlined', outlined: true),
          const SizedBox(height: IdrSpace.xl),
          const Row(
            children: [
              Expanded(
                child: RoundInfo(
                  icon: Icons.bolt,
                  line1: '220 kW',
                  line2: '34 min to 100%',
                ),
              ),
              Expanded(
                child: RoundInfo(
                  icon: Icons.local_gas_station_outlined,
                  line1: '2 of 3',
                  line2: 'available',
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _Swatch extends StatelessWidget {
  const _Swatch(this.color, this.hex);

  final Color color;
  final String hex;

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Padding(
        padding: const EdgeInsets.only(right: 8),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            AspectRatio(
              aspectRatio: 1,
              child: Container(
                decoration: BoxDecoration(
                  color: color,
                  borderRadius: BorderRadius.circular(IdrRadius.tile),
                  border: Border.all(color: IdrColors.outline),
                ),
              ),
            ),
            const SizedBox(height: 6),
            Text(hex, style: IdrText.micro),
          ],
        ),
      ),
    );
  }
}
