import { existsSync } from "node:fs";
import path from "node:path";

import type { NextConfig } from "next";

// A provided car model replaces the built-in one (see src/lib/carModel.ts).
const CAR_MODEL = "/models/car.glb";
const hasCarModel = existsSync(path.join(process.cwd(), "public", CAR_MODEL));

const nextConfig: NextConfig = {
  // The "N" badge sits on top of the page; errors still show without it.
  devIndicators: false,
  // Test builds can go elsewhere (NEXT_DIST_DIR=.next-verify) so they never
  // swap files out from under a running `npm start` / `npm run dev`.
  distDir: process.env.NEXT_DIST_DIR || ".next",
  env: {
    NEXT_PUBLIC_CAR_MODEL: hasCarModel ? CAR_MODEL : "",
  },
  // Serve the APK(s) in public/downloads/ as an Android package download, so
  // phones offer to install it instead of opening it as an unknown file.
  headers() {
    return [
      {
        source: "/downloads/:path*",
        headers: [
          { key: "Content-Type", value: "application/vnd.android.package-archive" },
          { key: "Content-Disposition", value: "attachment" },
          { key: "Cache-Control", value: "public, max-age=0, must-revalidate" },
        ],
      },
    ];
  },
};

export default nextConfig;
