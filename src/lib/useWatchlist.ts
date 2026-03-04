"use client";

import { useState, useEffect, useCallback } from "react";

const STORAGE_KEY = "sentinel-watchlist";

/**
 * Hook for managing a localStorage-based species watchlist.
 * Persists across sessions. No auth required.
 */
export function useWatchlist() {
  const [watchedIds, setWatchedIds] = useState<string[]>([]);

  // Load from localStorage on mount
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) setWatchedIds(parsed);
      }
    } catch {
      // Ignore malformed data
    }
  }, []);

  // Persist to localStorage whenever watchedIds changes
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(watchedIds));
    } catch {
      // Storage full or unavailable
    }
  }, [watchedIds]);

  const isWatched = useCallback(
    (id: string) => watchedIds.includes(id),
    [watchedIds],
  );

  const toggle = useCallback((id: string) => {
    setWatchedIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id],
    );
  }, []);

  const add = useCallback((id: string) => {
    setWatchedIds((prev) => (prev.includes(id) ? prev : [...prev, id]));
  }, []);

  const remove = useCallback((id: string) => {
    setWatchedIds((prev) => prev.filter((x) => x !== id));
  }, []);

  const count = watchedIds.length;

  return { watchedIds, isWatched, toggle, add, remove, count };
}
