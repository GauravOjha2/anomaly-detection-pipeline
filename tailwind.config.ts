import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        severity: {
          critical: "#ef4444",
          warning: "#f59e0b",
          info: "#3b82f6",
          normal: "#10b981",
        },
        radar: {
          green: "#3b82f6",
          greenDim: "rgba(59, 130, 246, 0.08)",
          greenGlow: "rgba(59, 130, 246, 0.2)",
          cyan: "#60a5fa",
          cyanDim: "rgba(96, 165, 250, 0.08)",
          background: "#09090b",
          grid: "rgba(59, 130, 246, 0.03)",
          ring: "rgba(59, 130, 246, 0.06)",
        },
      },
      fontFamily: {
        sans: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-geist-mono)", "monospace"],
      },
      animation: {
        "radar-sweep": "radarSweep 4s linear infinite",
        "radar-ping": "radarPing 2s ease-out infinite",
        "blink": "blink 1s ease-in-out infinite",
      },
      keyframes: {
        radarSweep: {
          "0%": { transform: "rotate(0deg)" },
          "100%": { transform: "rotate(360deg)" },
        },
        radarPing: {
          "0%": { transform: "scale(0)", opacity: "1" },
          "100%": { transform: "scale(2)", opacity: "0" },
        },
        blink: {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.3" },
        },
      },
      backdropBlur: {
        xs: "2px",
      },
    },
  },
  plugins: [],
};
export default config;
