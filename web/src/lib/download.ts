export const SOURCE_URL = "https://github.com/rayyanshaikh123/Dead_Reckoning";

// The Android build is attached to the repo's latest GitHub release as
// `idr.apk`; this link always points at the newest one. Override with
// NEXT_PUBLIC_APK_URL at build time to link elsewhere.
export const APK_URL =
  process.env.NEXT_PUBLIC_APK_URL || `${SOURCE_URL}/releases/latest/download/idr.apk`;
