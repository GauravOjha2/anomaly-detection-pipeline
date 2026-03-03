import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "Sentinel - Real-Time Anomaly Detection Pipeline",
  description:
    "AI-powered tourist safety monitoring system using ensemble ML models. Real-time GPS trajectory analysis, anomaly detection, and intelligent alert dispatching.",
  keywords: [
    "anomaly detection",
    "machine learning",
    "tourist safety",
    "real-time monitoring",
    "ensemble models",
    "IoT",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-[#0a0a1a] text-white noise-bg`}
      >
        <div className="radar-grid-bg fixed inset-0 -z-10" />
        {children}
      </body>
    </html>
  );
}
