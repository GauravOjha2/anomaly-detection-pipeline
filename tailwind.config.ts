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
          info: "#2dd4bf",
          normal: "#10b981",
        },
        radar: {
          green: "#2dd4bf",
          greenDim: "rgba(45, 212, 191, 0.08)",
          greenGlow: "rgba(45, 212, 191, 0.2)",
          cyan: "#5eead4",
          cyanDim: "rgba(94, 234, 212, 0.08)",
          background: "#09090b",
          grid: "rgba(45, 212, 191, 0.03)",
          ring: "rgba(45, 212, 191, 0.06)",
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
