"use client";

import { motion, AnimatePresence } from "framer-motion";
import { X, AlertTriangle, AlertCircle, Info, CheckCircle } from "lucide-react";
import { useToast, type Toast } from "@/lib/useToast";

const levelConfig: Record<Toast['level'], { icon: typeof AlertTriangle; bg: string; iconColor: string; titleColor: string }> = {
  critical: {
    icon: AlertTriangle,
    bg: "bg-red-500/15 border-red-500/30",
    iconColor: "text-red-400",
    titleColor: "text-red-300",
  },
  warning: {
    icon: AlertCircle,
    bg: "bg-amber-500/15 border-amber-500/30",
    iconColor: "text-amber-400",
    titleColor: "text-amber-300",
  },
  info: {
    icon: Info,
    bg: "bg-teal-500/15 border-teal-500/30",
    iconColor: "text-teal-400",
    titleColor: "text-teal-300",
  },
  success: {
    icon: CheckCircle,
    bg: "bg-emerald-500/15 border-emerald-500/30",
    iconColor: "text-emerald-400",
    titleColor: "text-emerald-300",
  },
};

export default function ToastNotification() {
  const { toasts, dismiss } = useToast();

  return (
    <div className="fixed top-20 right-4 z-[100] flex flex-col gap-2 max-w-sm w-full pointer-events-none">
      <AnimatePresence>
        {toasts.map((toast) => {
          const config = levelConfig[toast.level];
          const Icon = config.icon;

          return (
            <motion.div
              key={toast.id}
              initial={{ opacity: 0, x: 100, scale: 0.9 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 100, scale: 0.9 }}
              transition={{ type: "spring", stiffness: 400, damping: 30 }}
              className={`pointer-events-auto flex items-start gap-3 px-4 py-3 rounded-xl border backdrop-blur-xl ${config.bg}`}
            >
              <Icon className={`w-4 h-4 mt-0.5 shrink-0 ${config.iconColor}`} />
              <div className="flex-1 min-w-0">
                <p className={`text-sm font-medium ${config.titleColor}`}>
                  {toast.title}
                </p>
                {toast.description && (
                  <p className="text-xs text-zinc-400 mt-0.5 truncate">
                    {toast.description}
                  </p>
                )}
              </div>
              <button
                onClick={() => dismiss(toast.id)}
                className="shrink-0 p-0.5 text-zinc-500 hover:text-white transition-colors"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
