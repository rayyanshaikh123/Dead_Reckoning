// Where the Android build is served from: `public/downloads/idr.apk`, deployed
// with the site (the repo is private, so GitHub release assets would 404 for
// visitors). Override with NEXT_PUBLIC_APK_URL at build time to link elsewhere.
export const APK_URL = process.env.NEXT_PUBLIC_APK_URL || "/downloads/idr.apk";

export const SOURCE_URL = "https://github.com/MANASMORE/Dead_Reckoning";
