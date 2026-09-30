import type { Metadata, Viewport } from "next";
import { Sometype_Mono } from "next/font/google";
import "./globals.css";

const mono = Sometype_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

const title = "IDR — keeps navigating when GPS drops";
const description =
  "IDR is an on-device AI that reads your phone's motion sensors to keep tracking your car through tunnels, underpasses and city canyons when GPS drops.";

// The logo files next to this layout (favicon.ico, icon.png, apple-icon.png,
// opengraph-image.png, twitter-image.png) are picked up by Next.js on their
// own; they're generated from app/assets/icon/icon-1024.png.
export const metadata: Metadata = {
  title,
  description,
  applicationName: "IDR",
  openGraph: { type: "website", siteName: "IDR", title, description },
  twitter: { card: "summary_large_image", title, description },
};

export const viewport: Viewport = {
  themeColor: "#171818",
  colorScheme: "dark",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={mono.variable}>
      <body>{children}</body>
    </html>
  );
}
