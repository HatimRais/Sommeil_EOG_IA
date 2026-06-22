import { ThemeProvider } from "@/components/providers/ThemeProvider";
import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import "./mobile.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "DeepSleep AI · Polysomnographie",
  description:
    "Plateforme clinique d'analyse automatisée des stades du sommeil par IA (EOG) — AASM 5 classes",
  icons: { icon: "⚕️" },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fr" suppressHydrationWarning className={`${inter.variable} h-full`}>
      <body className="h-full overflow-hidden antialiased">
        <ThemeProvider>{children}</ThemeProvider>
      </body>
    </html>
  );
}
