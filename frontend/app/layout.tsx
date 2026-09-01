import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Exception Engine",
  description:
    "Learns what normal operational behaviour looks like, in context, and raises a case only when something meaningfully deviates.",
};

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/exceptions", label: "Exception inbox" },
  { href: "/schemas", label: "Schemas & baselines" },
  { href: "/policies", label: "Suppression policies" },
  { href: "/feedback", label: "Detector health" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <aside className="sidebar">
            <div className="brand">
              <span className="brand-mark" aria-hidden />
              <span>Exception Engine</span>
            </div>
            <nav>
              {NAV.map((item) => (
                <Link key={item.href} href={item.href}>
                  {item.label}
                </Link>
              ))}
            </nav>
            <p className="sidebar-note">
              Every score here comes from a deterministic ensemble and can be recomputed
              from the stored detector outputs. The narrative text only rephrases that
              evidence; it never sets a score.
            </p>
          </aside>
          <main className="content">{children}</main>
        </div>
      </body>
    </html>
  );
}
