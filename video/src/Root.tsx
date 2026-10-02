import { Composition } from 'remotion';
import { IDRStory } from './IDRStory';
import { DURATION_S, FPS } from './timeline';

export const RemotionRoot: React.FC = () => (
  <Composition id="IDRStory" component={IDRStory} durationInFrames={DURATION_S * FPS} fps={FPS} width={1920} height={1080} />
);
