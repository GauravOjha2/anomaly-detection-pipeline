"use client";

import dynamic from "next/dynamic";
import type { ComponentProps } from "react";
import type WildlifeMapInnerComponent from "./WildlifeMapInner";

type WildlifeMapProps = ComponentProps<typeof WildlifeMapInnerComponent>;

const WildlifeMapInner = dynamic(() => import("./WildlifeMapInner"), {
  ssr: false,
  loading: () => (
    <div
      className="flex items-center justify-center bg-zinc-900/50 rounded-xl border border-zinc-800"
      style={{ height: "500px" }}
    >
      <div className="flex flex-col items-center gap-2">
        <div className="w-6 h-6 rounded-full border-2 border-radar-green/30 border-t-radar-green animate-spin" />
        <span className="text-zinc-500 text-sm">Loading map...</span>
      </div>
    </div>
  ),
});

export default function WildlifeMap(props: WildlifeMapProps) {
  return <WildlifeMapInner {...props} />;
}
