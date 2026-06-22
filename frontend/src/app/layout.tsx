import { ThemeProvider } from "@/components/providers/ThemeProvider";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

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
