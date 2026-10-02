import type { SceneState } from './Overlays';

export function createEngine(canvas: HTMLCanvasElement | null, assetUrl: (path: string) => string): Promise<{
  MAP: { sens: Record<string, number[]>; vest: number[] };
  update: (t: number) => SceneState;
  draw: () => void;
  dispose: () => void;
}>;
