"use client";

import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import type { Alert, DetectionResult } from './types';
import { TRACKED_SPECIES, type TrackedSpecies } from './species-data';

interface SentinelContextType {
  alerts: Alert[];
  species: TrackedSpecies[];
  isLoading: boolean;
  lastFetched: Date | null;
  error: string | null;
  soundEnabled: boolean;
  connectionStatus: 'connected' | 'degraded' | 'offline';
  fetchAlerts: () => Promise<void>;
  setSoundEnabled: (enabled: boolean) => void;
  getAlertsBySpecies: (speciesId: string) => Alert[];
  getAlertsByLevel: (level: Alert['alert_level']) => Alert[];
  getCriticalAlerts: () => Alert[];
  getStats: () => {
    totalAlerts: number;
    criticalCount: number;
    warningCount: number;
    infoCount: number;
    speciesCount: number;
  };
}

const SentinelContext = createContext<SentinelContextType | undefined>(undefined);

const POLL_INTERVAL = 60000;

export function SentinelProvider({ children }: { children: React.ReactNode }) {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [lastFetched, setLastFetched] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [soundEnabled, setSoundEnabledState] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'degraded' | 'offline'>('connected');
  
  const prevAlertIdsRef = useRef<Set<string>>(new Set());
  const criticalSoundPlayedRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    const stored = localStorage.getItem('sentinel-sound-enabled');
    if (stored !== null) {
      setSoundEnabledState(stored === 'true');
    }
  }, []);

  const setSoundEnabled = useCallback((enabled: boolean) => {
    setSoundEnabledState(enabled);
    localStorage.setItem('sentinel-sound-enabled', String(enabled));
  }, []);

  const fetchAlerts = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const response = await fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'live' }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data: DetectionResult = await response.json();
      
      if (data.data_source === 'inaturalist_live') {
        setConnectionStatus('connected');
      } else {
        setConnectionStatus('degraded');
      }

      const newAlerts = data.alerts || [];
      const newAlertIds = new Set(newAlerts.map(a => a.alert_id));
      const prevIds = prevAlertIdsRef.current;
      
      const criticalAlerts = newAlerts.filter(a => a.alert_level === 'CRITICAL');
      for (const alert of criticalAlerts) {
        if (!prevIds.has(alert.alert_id) && !criticalSoundPlayedRef.current.has(alert.alert_id)) {
          if (soundEnabled) {
            playAlertSound();
          }
          criticalSoundPlayedRef.current.add(alert.alert_id);
        }
      }
      
      const currentIds = Array.from(criticalSoundPlayedRef.current);
      criticalSoundPlayedRef.current = new Set(
        currentIds.filter(id => newAlertIds.has(id))
      );
      
      prevAlertIdsRef.current = newAlertIds;
      
      setAlerts(newAlerts);
      setLastFetched(new Date());
    } catch (err) {
      setConnectionStatus('offline');
      setError(err instanceof Error ? err.message : 'Failed to fetch alerts');
    } finally {
      setIsLoading(false);
    }
  }, [soundEnabled]);

  useEffect(() => {
    fetchAlerts();
    const interval = setInterval(fetchAlerts, POLL_INTERVAL);
    return () => clearInterval(interval);
  }, [fetchAlerts]);

  const getAlertsBySpecies = useCallback((speciesId: string) => {
    return alerts.filter(a => a.species_name.toLowerCase().replace(/ /g, '-') === speciesId.toLowerCase());
  }, [alerts]);

  const getAlertsByLevel = useCallback((level: Alert['alert_level']) => {
    return alerts.filter(a => a.alert_level === level);
  }, [alerts]);

  const getCriticalAlerts = useCallback(() => {
    return alerts.filter(a => a.alert_level === 'CRITICAL');
  }, [alerts]);

  const getStats = useCallback(() => {
    return {
      totalAlerts: alerts.length,
      criticalCount: alerts.filter(a => a.alert_level === 'CRITICAL').length,
      warningCount: alerts.filter(a => a.alert_level === 'WARNING').length,
      infoCount: alerts.filter(a => a.alert_level === 'INFO').length,
      speciesCount: TRACKED_SPECIES.length,
    };
  }, [alerts]);

  return (
    <SentinelContext.Provider
      value={{
        alerts,
        species: TRACKED_SPECIES,
        isLoading,
        lastFetched,
        error,
        soundEnabled,
        connectionStatus,
        fetchAlerts,
        setSoundEnabled,
        getAlertsBySpecies,
        getAlertsByLevel,
        getCriticalAlerts,
        getStats,
      }}
    >
      {children}
    </SentinelContext.Provider>
  );
}

export function useSentinel() {
  const context = useContext(SentinelContext);
  if (context === undefined) {
    throw new Error('useSentinel must be used within a SentinelProvider');
  }
  return context;
}

function playAlertSound() {
  try {
    const audioContext = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.setValueAtTime(880, audioContext.currentTime);
    oscillator.type = 'sine';
    
    gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.3);
  } catch {
    // Audio not supported or blocked
  }
}
