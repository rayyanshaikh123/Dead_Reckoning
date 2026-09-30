// Where the Android build is served from. Put the APK at
// `public/downloads/idr.apk`, or point NEXT_PUBLIC_APK_URL at a release
// (e.g. a GitHub release asset) at build time.
export const APK_URL = process.env.NEXT_PUBLIC_APK_URL || "/downloads/idr.apk";

export const SOURCE_URL = "https://github.com/MANASMORE/Dead_Reckoning";
