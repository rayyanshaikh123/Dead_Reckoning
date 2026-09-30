import type { Metadata, Viewport } from "next";
import { Sometype_Mono } from "next/font/google";
import "./globals.css";

const mono = Sometype_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "IDR — keeps navigating when GPS drops",
  description:
    "IDR is an on-device AI that reads your phone's motion sensors to keep tracking your car through tunnels, underpasses and city canyons when GPS drops.",
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
