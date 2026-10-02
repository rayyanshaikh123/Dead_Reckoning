import '@fontsource/sometype-mono/400.css';
import '@fontsource/sometype-mono/500.css';
import '@fontsource/sometype-mono/600.css';
import '@fontsource/sometype-mono/700.css';
import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { AbsoluteFill, Audio, Sequence, continueRender, delayRender, staticFile, useCurrentFrame, useVideoConfig } from 'remotion';
import { createEngine } from './engine';
import { Overlays, SceneState } from './Overlays';
import { VO, duck } from './timeline';

type Engine = Awaited<ReturnType<typeof createEngine>>;

export const IDRStory: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();
  const t = frame / fps;
  const canvas = useRef<HTMLCanvasElement>(null);
  const [engine, setEngine] = useState<Engine | null>(null);
  const [handle] = useState(() => delayRender('Building the 3D scenes', { timeoutInMilliseconds: 180000 }));

  useEffect(() => {
    let alive = true, made: Engine | null = null;
    Promise.all([createEngine(canvas.current, (p: string) => staticFile(p)), document.fonts.ready]).then(([e]) => {
      made = e;
      if (!alive) { e.dispose(); return; }
      setEngine(e); continueRender(handle);
    });
    return () => { alive = false; made?.dispose(); };
  }, [handle]);

  // Pose the 3D scene for this frame (also gives the overlays their screen anchors), then draw it.
  const state = useMemo(() => (engine ? engine.update(t) : null), [engine, t]);
  useLayoutEffect(() => { if (engine && state) engine.draw(); }, [engine, state]);

  return (
    <AbsoluteFill style={{ background: '#000' }}>
      <canvas ref={canvas} width={width} height={height} style={{ position: 'absolute', left: 0, top: 0, width, height }} />
      {state && engine && <Overlays t={t} s={state} map={engine.MAP} />}
      <Audio src={staticFile('audio/score.wav')} volume={(f) => 1 - 0.62 * duck(f / fps)} />
      {VO.map(([name, t0]) => (
        <Sequence key={name} from={Math.round(t0 * fps)} layout="none">
          <Audio src={staticFile(`audio/${name}.wav`)} />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};
