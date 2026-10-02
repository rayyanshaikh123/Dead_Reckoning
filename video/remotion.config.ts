import { Config } from '@remotion/cli/config';

// The 3D scenes use WebGL. 'angle' renders on the GPU; on a machine without one, pass --gl=swiftshader.
Config.setChromiumOpenGlRenderer('angle');
Config.setVideoImageFormat('jpeg');
Config.setJpegQuality(95);
Config.setCodec('h264');
Config.setCrf(18);
Config.setPixelFormat('yuv420p');
