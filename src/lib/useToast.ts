"use client";

import { useState, useCallback, useRef } from "react";

export interface Toast {
  id: string;
  title: string;
  description?: string;
  level: "critical" | "warning" | "info" | "success";
  timestamp: number;
}

let _toastId = 0;

/**
 * Hook for managing toast notifications.
 * Toasts auto-dismiss after a configurable duration.
 */
export function useToast(autoDismissMs = 6000) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const timersRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map());

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
    const timer = timersRef.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timersRef.current.delete(id);
    }
  }, []);

  const addToast = useCallback(
    (toast: Omit<Toast, "id" | "timestamp">) => {
      const id = `toast-${++_toastId}`;
      const newToast: Toast = { ...toast, id, timestamp: Date.now() };
      setToasts((prev) => [newToast, ...prev].slice(0, 5)); // Keep max 5

      // Auto-dismiss
      const timer = setTimeout(() => dismiss(id), autoDismissMs);
      timersRef.current.set(id, timer);

      return id;
    },
    [autoDismissMs, dismiss],
  );

  const clearAll = useCallback(() => {
    timersRef.current.forEach((timer) => clearTimeout(timer));
    timersRef.current.clear();
    setToasts([]);
  }, []);

  return { toasts, addToast, dismiss, clearAll };
}
