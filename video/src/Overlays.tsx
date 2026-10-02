import React from 'react';
import { AbsoluteFill } from 'remotion';
import { clamp, ei, env, eo, eio, lerp, lin } from './timeline';

// Brand tokens (same as the app and website)
const ACCENT = '#f06131';
const TEXT = '#f2f2f2';
const TEXT2 = '#8c8d8d';
const BLUE = '#4285f4';
const GREEN = '#48fa5d';
const RED = '#e5484d';
const FONT = '"Sometype Mono", ui-monospace, monospace';

export const MARK = 'M70 352 C150 352 175 300 205 270 C240 236 270 250 300 222 C330 194 360 168 440 150';
export const MARK_TUNNEL = 'M205 270 C240 236 270 250 300 222';

type Pt = [number, number, number];
export type SceneState = {
  useMap: boolean;
  fade: number;
  gps: Pt; ins: Pt; idr: Pt; tunMid: Pt;
  insOff: number;
  replay: { seconds: number; speedKmh: number; distanceM: number };
  sensorCursor: number;
};
type MapData = { sens: Record<string, number[]>; vest: number[] };

// ---------- building blocks ----------
const Layer: React.FC<{ o: number; style?: React.CSSProperties; children: React.ReactNode }> = ({ o, style, children }) =>
  o <= 0.001 ? null : <div style={{ position: 'absolute', opacity: o, ...style }}>{children}</div>;

/** Text that rises in letter by letter, each letter un-blurring as it lands. */
const Kinetic: React.FC<{ text: string; t: number; t0: number; stagger?: number; dur?: number; dy?: number; blur?: number; style?: React.CSSProperties }> =
  ({ text, t, t0, stagger = 0.03, dur = 0.35, dy = 60, blur = 12, style }) => (
    <div style={style}>
      {[...text].map((c, i) => {
        const k = eo(lin(t, t0 + i * stagger, t0 + i * stagger + dur));
        return <span key={i} style={{ display: 'inline-block', whiteSpace: 'pre', opacity: k, transform: `translateY(${(1 - k) * dy}px)`, filter: k < 1 ? `blur(${(1 - k) * blur}px)` : undefined }}>{c}</span>;
      })}
    </div>
  );

const Kicker: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) =>
  <div style={{ fontSize: 22, letterSpacing: '.32em', color: ACCENT, fontWeight: 500, ...style }}>{children}</div>;

const Chip: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) =>
  <div style={{ display: 'inline-block', border: `2px solid ${ACCENT}`, color: TEXT, fontSize: 24, letterSpacing: '.22em', padding: '12px 22px', background: 'rgba(15,16,16,.6)', ...style }}>{children}</div>;

const Tag: React.FC<{ color: string; children: React.ReactNode; style?: React.CSSProperties }> = ({ color, children, style }) =>
  <div style={{ fontSize: 19, letterSpacing: '.16em', textTransform: 'uppercase', whiteSpace: 'nowrap', padding: '8px 14px', background: 'rgba(10,10,10,.7)', borderLeft: `3px solid ${color}`, ...style }}>{children}</div>;

const Hud: React.FC<{ label: string; value: React.ReactNode; color?: string }> = ({ label, value, color = TEXT }) =>
  <div style={{ fontSize: 20, letterSpacing: '.14em', color: TEXT2, lineHeight: 1.9 }}>{label}&nbsp;&nbsp;<b style={{ color, fontWeight: 500, letterSpacing: '.06em', fontSize: 30 }}>{value}</b></div>;

const anchor = (p: Pt, dx: number, dy: number): React.CSSProperties => ({ left: 0, top: 0, transform: `translate(${p[0] + dx}px, ${p[1] + dy}px)` });

// ---------- sections ----------
const Intro: React.FC<{ t: number }> = ({ t }) => (
  <Layer o={env(t, 1.5, 1.7, 4.6, 4.9)} style={{ left: 150, top: 510 }}>
    <Kinetic text="Every map app has a blind spot." t={t} t0={1.5} dy={24} blur={10} style={{ fontSize: 34, letterSpacing: '.12em', fontWeight: 500 }} />
  </Layer>
);

const SATS: [number, number][] = [[560, 70], [1180, 40], [1640, 110]];
const Problem: React.FC<{ t: number; s: SceneState }> = ({ t, s }) => {
  const words: [string, number, number][] = [['Tunnels.', 5.25, 5.95], ['Underpasses.', 5.95, 6.75], ['City canyons.', 6.75, 7.65]];
  const lost = t > 7.95;
  const satO = env(t, 5.1, 5.5, 9.6, 10.2);
  const blink = t < 9.2 ? (Math.floor((t - 7.95) * 6) % 2 === 0 ? 1 : 0.4) : 1;
  return (
    <>
      {words.map(([w, a, b]) => (
        <Layer key={w} o={env(t, a, a + 0.05, b - 0.05, b + 0.02)} style={{ left: 130, top: 700 }}>
          <Kicker style={{ marginBottom: 14 }}>THE PROBLEM</Kicker>
          <Kinetic text={w} t={t} t0={a} stagger={0.02} dur={0.22} dy={70} blur={14} style={{ fontSize: 110, fontWeight: 600, lineHeight: 1 }} />
        </Layer>
      ))}
      {satO > 0 && (
        <svg width={1920} height={1080} style={{ position: 'absolute', left: 0, top: 0, opacity: satO, overflow: 'visible' }}>
          {SATS.map(([x, y], i) => {
            const cut = lost ? 1 - eo(lin(t, 7.95 + i * 0.06, 8.3 + i * 0.06)) : 1;
            const xo = lost ? eo(lin(t, 8.0 + i * 0.08, 8.25 + i * 0.08)) : 0;
            return (
              <g key={i}>
                <line x1={x} y1={y + 14} x2={lerp(x, s.gps[0], cut)} y2={lerp(y + 14, s.gps[1], cut)} stroke={BLUE} strokeWidth={2.5} strokeDasharray="10 12" strokeDashoffset={-t * 60} opacity={lost ? 0.6 : 0.85} />
                <g transform={`translate(${x},${y})`} opacity={lost ? 0.55 : 1}>
                  <rect x={-14} y={-10} width={28} height={20} rx={4} fill={TEXT} />
                  <rect x={-46} y={-6} width={28} height={12} fill={BLUE} /><rect x={18} y={-6} width={28} height={12} fill={BLUE} />
                  <g stroke={ACCENT} strokeWidth={5} strokeLinecap="round" opacity={xo}><line x1={-22} y1={-22} x2={22} y2={22} /><line x1={22} y1={-22} x2={-22} y2={22} /></g>
                </g>
              </g>
            );
          })}
        </svg>
      )}
      <Layer o={env(t, 7.95, 8.05, 13.7, 13.95) * blink} style={{ left: 130, top: 150, transform: `scale(${1 + 0.15 * (1 - eo(lin(t, 7.95, 8.25)))})`, transformOrigin: 'left center' }}>
        <Chip style={{ fontSize: 30, padding: '16px 28px' }}>GPS SIGNAL LOST</Chip>
      </Layer>
      {t > 8.3 && <Layer o={env(t, 8.4, 8.7, 13.7, 13.95)} style={anchor(s.gps, 40, -20)}><Tag color={BLUE}>Your map app · <span style={{ color: BLUE }}>frozen</span></Tag></Layer>}
      {t > 10.6 && <Layer o={env(t, 10.7, 11.0, 13.7, 13.95)} style={anchor(s.ins, 40, -20)}><Tag color={RED}>Phone sensors + plain math · <span style={{ color: RED }}>{s.insOff} m off</span></Tag></Layer>}
    </>
  );
};

const STEPS = [
  { a: 14.0, b: 17.0, k: 'HOW IT WORKS · 01', title: 'Sense.', sub: ['Accelerometer · gyroscope · magnetometer.', 'The sensors already in your phone.'] },
  { a: 17.0, b: 21.3, k: 'HOW IT WORKS · 02', title: 'Think.', sub: ['An on-device AI (IDNN v5) estimates speed', 'and heading 10× a second. No internet.'] },
  { a: 21.3, b: 25.6, k: 'HOW IT WORKS · 03', title: 'Stay on the road.', sub: ['Position is matched to OpenStreetMap roads,', 'cached offline on the phone.'] },
];
const Steps: React.FC<{ t: number }> = ({ t }) => (
  <>
    {STEPS.map(({ a, b, k, title, sub }) => {
      const sk = eo(lin(t, a + 0.4, a + 0.8));
      return (
        <Layer key={k} o={env(t, a + 0.05, a + 0.15, b - 0.1, b)} style={{ left: 130, bottom: 120, transform: `translateX(${-ei(lin(t, b - 0.16, b)) * 240}px)` }}>
          <Kicker style={{ marginBottom: 18 }}>{k}</Kicker>
          <Kinetic text={title} t={t} t0={a + 0.1} stagger={0.025} dur={0.3} dy={90} blur={16} style={{ fontSize: 110, fontWeight: 600, lineHeight: 1 }} />
          <div style={{ height: 3, background: ACCENT, width: eo(lin(t, a + 0.2, a + 0.7)) * 140, margin: '24px 0 22px' }} />
          <div style={{ fontSize: 27, color: '#b9baba', letterSpacing: '.04em', opacity: sk, transform: `translateY(${(1 - sk) * 16}px)` }}>{sub[0]}<br />{sub[1]}</div>
        </Layer>
      );
    })}
    {[23.0, 23.85, 24.7].map((a, i) => (
      <Layer key={a} o={env(t, a, a + 0.12, 25.4, 25.6)} style={{ right: 130, top: 180 + i * 86, transform: `translateX(${(1 - eo(lin(t, a, a + 0.3))) * 60}px)` }}>
        <Chip>{['NO GPS', 'NO INTERNET', 'NO EXTRA HARDWARE'][i]}</Chip>
      </Layer>
    ))}
  </>
);

// "Think" panel: the real sensor stream from the drive, a small network, the speed estimate
const NNL = [6, 9, 9, 1], NNX = [40, 260, 480, 700];
const NODES = NNL.flatMap((n, l) => Array.from({ length: n }, (_, i) => ({ l, x: NNX[l], y: 65 + (i - (n - 1) / 2) * 13 })));
const EDGES = NODES.flatMap(a => NODES.filter(b => b.l === a.l + 1).map(b => [a, b] as const));
const SENS: [string, string, string][] = [['ACCEL X', 'ax', ACCENT], ['ACCEL Y', 'ay', GREEN], ['GYRO Z', 'gz', BLUE]];
const ThinkPanel: React.FC<{ t: number; s: SceneState; map: MapData }> = ({ t, s, map }) => {
  const o = env(t, 17.15, 17.5, 21.1, 21.3);
  if (o <= 0) return null;
  const cur = s.sensorCursor;
  const pulse = (t * 2.5) % 1;
  const hash = (i: number) => { const x = Math.sin(i * 12.9898 + Math.floor(t * 10)) * 43758.5453; return x - Math.floor(x); };
  return (
    <Layer o={o} style={{ left: 120, top: 50, transform: `translateY(${(1 - eo(lin(t, 17.15, 17.6))) * 30}px)` }}>
      <div style={{ width: 820, background: 'rgba(23,24,24,.78)', border: '1px solid #2e2f2f', borderRadius: 22, padding: '30px 36px', boxSizing: 'border-box' }}>
        <div style={{ fontSize: 20, letterSpacing: '.14em', color: TEXT2, marginBottom: 8 }}>LIVE SENSOR STREAM · 10 Hz</div>
        <svg width={748} height={220} style={{ display: 'block' }}>
          {SENS.map(([name, key, col], r) => {
            const y0 = 42 + r * 72; const data = map.sens[key];
            let d = '';
            for (let x = 0; x <= 748; x += 4) {
              const si = cur - 120 + (x / 748) * 120; const i = Math.floor(si);
              const v = lerp(data[i] ?? 0, data[i + 1] ?? 0, si - i);
              d += `${x ? 'L' : 'M'}${x} ${(y0 - clamp(v, -2.2, 2.2) * 11).toFixed(1)}`;
            }
            return (
              <g key={key}>
                <text x={0} y={y0 - 22} fill="#5c5d5d" fontSize={18} fontFamily={FONT}>{name}</text>
                <line x1={0} x2={748} y1={y0} y2={y0} stroke="#2e2f2f" />
                <path d={d} fill="none" stroke={col} strokeWidth={2.5} />
              </g>
            );
          })}
        </svg>
        <svg width={748} height={130} style={{ display: 'block', marginTop: 6 }}>
          {EDGES.map(([a, b], i) => (
            <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={ACCENT} strokeWidth={1}
              opacity={0.08 + 0.5 * Math.max(0, 1 - Math.abs(pulse * 3 - a.l - 0.5) * 2) * (0.4 + 0.6 * hash(i))} />
          ))}
          {NODES.map((n, i) => (
            <circle key={i} cx={n.x} cy={n.y} r={n.l === 3 ? 9 : 5.5} fill={n.l === 3 ? ACCENT : TEXT}
              opacity={0.35 + 0.65 * Math.max(0, 1 - Math.abs(pulse * 3 - n.l + 0.2) * 1.5)} />
          ))}
        </svg>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginTop: 10 }}>
          <div style={{ fontSize: 20, letterSpacing: '.14em', color: TEXT2, lineHeight: 1.6 }}>ESTIMATED SPEED<br />
            <b style={{ fontSize: 60, color: ACCENT, fontWeight: 500 }}>{Math.round(map.vest[Math.floor(cur)] ?? 0)}</b> <span style={{ fontSize: 24, color: TEXT }}>km/h</span></div>
          <div style={{ fontSize: 20, letterSpacing: '.14em', color: TEXT2, lineHeight: 1.6, textAlign: 'right' }}>GPS<br /><b style={{ color: ACCENT, fontSize: 30, fontWeight: 500 }}>OFF</b></div>
        </div>
      </div>
    </Layer>
  );
};

const Dot: React.FC<{ c: string; hollow?: boolean }> = ({ c, hollow }) =>
  <span style={{ display: 'inline-block', width: hollow ? 12 : 16, height: hollow ? 12 : 16, borderRadius: '50%', background: hollow ? 'none' : c, border: hollow ? `3px solid ${c}` : undefined, marginRight: 14, verticalAlign: -2 }} />;

const UseCase: React.FC<{ t: number; s: SceneState }> = ({ t, s }) => {
  const hO = env(t, 25.9, 26.3, 30.8, 31.0);
  const rO = env(t, 31.3, 31.6, 34.75, 35.0);
  const k2 = eo(lin(t, 33.0, 33.5));
  return (
    <>
      <Layer o={hO} style={{ left: 130, top: 120 }}>
        <Kicker style={{ marginBottom: 16 }}>TUNNEL TEST · REAL DRIVE</Kicker>
        <Hud label="GPS" value="OFF" color={ACCENT} />
        <Hud label="TIME WITHOUT GPS" value={`${s.replay.seconds.toFixed(1)} s`} />
        <Hud label="IDR SPEED" value={`${s.replay.speedKmh} km/h`} />
        <Hud label="DISTANCE" value={`${s.replay.distanceM} m`} />
      </Layer>
      <Layer o={hO} style={{ right: 130, top: 130, fontSize: 20, letterSpacing: '.14em', color: TEXT2, lineHeight: 2.1 }}>
        <div><Dot c={ACCENT} />IDR</div>
        <div><Dot c="#fff" hollow />TRUE POSITION</div>
        <div><Dot c={RED} />PHONE SENSORS + PLAIN MATH</div>
      </Layer>
      {t > 26 && t < 31 && <Layer o={env(t, 26.3, 26.6, 30.6, 30.9)} style={anchor(s.tunMid, -180, 50)}><Tag color={ACCENT}>Tunnel · 648 m · 30 s without GPS</Tag></Layer>}
      {t > 31.6 && t < 35 && <Layer o={env(t, 31.8, 32.1, 34.7, 34.95)} style={anchor(s.idr, 60, -90)}><Tag color={ACCENT} style={{ fontSize: 24 }}>IDR exit error · <span style={{ color: ACCENT }}>14 m</span></Tag></Layer>}
      <Layer o={rO} style={{ left: 130, bottom: 120, transform: `translateX(${-ei(lin(t, 34.84, 35.0)) * 240}px)` }}>
        <Kicker style={{ marginBottom: 18 }}>RESULT</Kicker>
        <div style={{ fontSize: 64, fontWeight: 600, lineHeight: 1.05 }}><span style={{ color: ACCENT }}>14 m</span> off after 648 m</div>
        <div style={{ fontSize: 27, color: '#b9baba', letterSpacing: '.04em', marginTop: 18 }}>IDR exit error 2.2% &nbsp;·&nbsp; plain sensor math: 792 m off</div>
        <div style={{ height: 3, background: ACCENT, width: eo(lin(t, 32.4, 33.0)) * 160, margin: '28px 0 26px' }} />
        <div style={{ fontSize: 44, fontWeight: 600, lineHeight: 1.05, opacity: k2, transform: `translateY(${(1 - k2) * 20}px)` }}>All 5 benchmark tunnels under<br />the 10% drift target.</div>
      </Layer>
    </>
  );
};

const Outro: React.FC<{ t: number; fade: number }> = ({ t, fade }) => {
  if (t <= 36.5) return null;
  const draw = eio(lin(t, 36.5, 37.4)), seg = eo(lin(t, 37.0, 37.5)), dot = eo(lin(t, 37.25, 37.55)), sk = eo(lin(t, 38.2, 38.8));
  return (
    <Layer o={fade} style={{ left: 1010, top: 250 }}>
      <svg width={220} height={220} viewBox="0 0 512 512" style={{ display: 'block' }}>
        <path d={MARK} fill="none" stroke={TEXT} strokeWidth={36} strokeLinecap="round" pathLength={1} strokeDasharray={1} strokeDashoffset={1 - draw} />
        <path d={MARK_TUNNEL} fill="none" stroke={ACCENT} strokeWidth={36} strokeLinecap="round" pathLength={1} strokeDasharray={1} strokeDashoffset={1 - seg} />
        <circle cx={440} cy={150} r={36 * dot} fill={TEXT} />
      </svg>
      <div style={{ display: 'flex', alignItems: 'baseline', marginTop: 10 }}>
        <Kinetic text="IDR" t={t} t0={36.6} stagger={0.07} dur={0.45} dy={80} blur={20} style={{ fontSize: 150, fontWeight: 700, letterSpacing: '.06em', lineHeight: 1 }} />
        <span style={{ fontSize: 150, fontWeight: 700, color: ACCENT, lineHeight: 1, opacity: eo(lin(t, 36.81, 37.26)) }}>.</span>
      </div>
      <Kinetic text="Lose the signal. Not the road." t={t} t0={37.3} stagger={0.022} dur={0.3} dy={24} blur={10} style={{ fontSize: 40, fontWeight: 500, letterSpacing: '.04em', marginTop: 30 }} />
      <div style={{ fontSize: 20, letterSpacing: `${0.32 + (1 - sk) * 0.3}em`, color: TEXT2, marginTop: 26, opacity: sk }}>INTELLIGENT DEAD RECKONING · ANDROID</div>
    </Layer>
  );
};

export const Overlays: React.FC<{ t: number; s: SceneState; map: MapData }> = ({ t, s, map }) => (
  <AbsoluteFill style={{ fontFamily: FONT, color: TEXT, overflow: 'hidden' }}>
    <Intro t={t} />
    {t >= 5 && t < 14 && <Problem t={t} s={s} />}
    <Steps t={t} />
    <ThinkPanel t={t} s={s} map={map} />
    {t >= 25 && t < 35 && <UseCase t={t} s={s} />}
    <Outro t={t} fade={s.fade} />
  </AbsoluteFill>
);
