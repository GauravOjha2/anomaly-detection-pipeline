"use client";

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
} from "react";
import type { TrackedAnimal } from "./movebank";
import { getAllTrackedAnimals, getTrackedAnimals, TRACKED_STUDIES } from "./movebank";

interface AnimalTrackingContextType {
  animals: TrackedAnimal[];
  isLoading: boolean;
  lastFetched: Date | null;
  error: string | null;
  followedAnimals: Set<string>; // animal IDs the user follows
  followAnimal: (id: string) => void;
  unfollowAnimal: (id: string) => void;
  isFollowing: (id: string) => boolean;
  getAnimalsByStudy: (studyId: number) => TrackedAnimal[];
  getAnimal: (id: string) => TrackedAnimal | undefined;
  refreshAnimals: () => Promise<void>;
  refreshStudy: (studyId: number) => Promise<void>;
  studies: typeof TRACKED_STUDIES;
}

const AnimalTrackingContext = createContext<
  AnimalTrackingContextType | undefined
>(undefined);

const ANIMAL_POLL_INTERVAL = 5 * 60 * 1000; // 5 minutes
const FOLLOWED_KEY = "sentinel-followed-animals";

export function AnimalTrackingProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [animals, setAnimals] = useState<TrackedAnimal[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [lastFetched, setLastFetched] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [followedAnimals, setFollowedAnimals] = useState<Set<string>>(
    new Set()
  );
  const initialLoadDone = useRef(false);

  // Load followed animals from localStorage
  useEffect(() => {
    try {
      const stored = localStorage.getItem(FOLLOWED_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as string[];
        setFollowedAnimals(new Set(parsed));
      }
    } catch {
      // Ignore parse errors
    }
  }, []);

  const persistFollowed = useCallback((ids: Set<string>) => {
    localStorage.setItem(FOLLOWED_KEY, JSON.stringify(Array.from(ids)));
  }, []);

  const followAnimal = useCallback(
    (id: string) => {
      setFollowedAnimals((prev) => {
        const next = new Set(prev);
        next.add(id);
        persistFollowed(next);
        return next;
      });
    },
    [persistFollowed]
  );

  const unfollowAnimal = useCallback(
    (id: string) => {
      setFollowedAnimals((prev) => {
        const next = new Set(prev);
        next.delete(id);
        persistFollowed(next);
        return next;
      });
    },
    [persistFollowed]
  );

  const isFollowing = useCallback(
    (id: string) => followedAnimals.has(id),
    [followedAnimals]
  );

  const refreshAnimals = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await getAllTrackedAnimals();
      setAnimals(data);
      setLastFetched(new Date());
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to fetch animal data"
      );
    } finally {
      setIsLoading(false);
    }
  }, []);

  const refreshStudy = useCallback(
    async (studyId: number) => {
      try {
        const studyAnimals = await getTrackedAnimals(studyId, 50);
        setAnimals((prev) => {
          // Replace animals from this study, keep the rest
          const filtered = prev.filter((a) => a.studyId !== studyId);
          return [...filtered, ...studyAnimals];
        });
        setLastFetched(new Date());
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Failed to refresh study"
        );
      }
    },
    []
  );

  const getAnimalsByStudy = useCallback(
    (studyId: number) => animals.filter((a) => a.studyId === studyId),
    [animals]
  );

  const getAnimal = useCallback(
    (id: string) => animals.find((a) => a.id === id),
    [animals]
  );

  // Initial fetch + polling
  useEffect(() => {
    if (!initialLoadDone.current) {
      initialLoadDone.current = true;
      refreshAnimals();
    }
    const interval = setInterval(refreshAnimals, ANIMAL_POLL_INTERVAL);
    return () => clearInterval(interval);
  }, [refreshAnimals]);

  return (
    <AnimalTrackingContext.Provider
      value={{
        animals,
        isLoading,
        lastFetched,
        error,
        followedAnimals,
        followAnimal,
        unfollowAnimal,
        isFollowing,
        getAnimalsByStudy,
        getAnimal,
        refreshAnimals,
        refreshStudy,
        studies: TRACKED_STUDIES,
      }}
    >
      {children}
    </AnimalTrackingContext.Provider>
  );
}

export function useAnimalTracking() {
  const context = useContext(AnimalTrackingContext);
  if (context === undefined) {
    throw new Error(
      "useAnimalTracking must be used within an AnimalTrackingProvider"
    );
  }
  return context;
}
