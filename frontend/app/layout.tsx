import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "PCB Flaw Detection",
  description: "Created by Slippery Seamen",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
}
