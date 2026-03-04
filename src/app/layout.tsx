import type { Metadata } from "next";
import localFont from "next/font/local";
import { SentinelProvider } from "@/lib/SentinelContext";
import { AnimalTrackingProvider } from "@/lib/AnimalTrackingContext";
import AuthProvider from "@/lib/AuthProvider";
import ToastNotification from "@/components/ToastNotification";
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
  title: "Sentinel — Wildlife Conservation Monitoring",
  description:
    "Real-time wildlife anomaly detection and conservation monitoring. Track endangered species, detect range anomalies, and protect critical habitats worldwide.",
  keywords: [
    "wildlife conservation",
    "endangered species",
    "anomaly detection",
    "species monitoring",
    "habitat protection",
    "conservation technology",
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
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-[#09090b] text-white noise-bg`}
      >
        <div className="radar-grid-bg fixed inset-0 -z-10" />
        <AuthProvider>
          <SentinelProvider>
            <AnimalTrackingProvider>
              {children}
              <ToastNotification />
            </AnimalTrackingProvider>
          </SentinelProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
