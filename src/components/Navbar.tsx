"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSession, signIn, signOut } from "next-auth/react";
import { motion, AnimatePresence } from "framer-motion";
import {
  LayoutDashboard,
  PawPrint,
  Bell,
  Menu,
  X,
  Shield,
  Search,
  Heart,
  Volume2,
  VolumeX,
  Wifi,
  WifiOff,
  Locate,
  LogIn,
  LogOut,
  User,
  ChevronDown,
} from "lucide-react";
import SearchCommand from "./SearchCommand";
import { useSentinel } from "@/lib/SentinelContext";
import { useWatchlist } from "@/lib/useWatchlist";
import { getTimeSince } from "@/lib/time";

const navLinks = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/species", label: "Species", icon: PawPrint },
  { href: "/animals", label: "Animals", icon: Locate },
  { href: "/alerts", label: "Alerts", icon: Bell },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [userMenuOpen, setUserMenuOpen] = useState(false);
  const userMenuRef = useRef<HTMLDivElement>(null);
  const pathname = usePathname();
  const { data: session } = useSession();
  
  const { alerts, connectionStatus, soundEnabled, setSoundEnabled, lastFetched } = useSentinel();
  const { count: watchlistCount } = useWatchlist();

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setSearchOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Close user menu on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (userMenuRef.current && !userMenuRef.current.contains(e.target as Node)) {
        setUserMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const isActive = (href: string) => pathname.startsWith(href);
  
  const criticalCount = alerts.filter(a => a.alert_level === 'CRITICAL').length;

  const ConnectionIcon = connectionStatus === 'connected' ? Wifi : connectionStatus === 'degraded' ? Wifi : WifiOff;
  const connectionColor = connectionStatus === 'connected' ? 'text-radar-green' : connectionStatus === 'degraded' ? 'text-yellow-500' : 'text-red-500';

  return (
    <>
      <motion.nav
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
          scrolled
            ? "bg-[#09090b]/80 backdrop-blur-xl border-b border-radar-green/10"
            : "bg-transparent"
        }`}
      >
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 group">
            <div className="w-8 h-8 rounded-lg bg-radar-greenDim flex items-center justify-center border border-radar-green/30 group-hover:shadow-lg group-hover:shadow-radar-green/20 transition-shadow">
              <Shield className="w-4 h-4 text-radar-green" />
            </div>
            <span className="font-semibold text-white text-sm tracking-tight">
              sentinel
            </span>
          </Link>

          {/* Desktop links + search + controls */}
          <div className="hidden md:flex items-center gap-1">
            {navLinks.map((link) => {
              const active = isActive(link.href);
              const Icon = link.icon;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`flex items-center gap-2 px-4 py-2 text-sm rounded-lg transition-colors ${
                    active
                      ? "text-radar-green bg-radar-green/10 border border-radar-green/20"
                      : "text-zinc-400 hover:text-white hover:bg-white/[0.04]"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {link.label}
                  {link.href === "/alerts" && criticalCount > 0 && (
                    <span className="ml-1 px-1.5 py-0.5 text-[10px] bg-red-500/20 text-red-400 rounded-full">
                      {criticalCount}
                    </span>
                  )}
                </Link>
              );
            })}

            {/* Divider */}
            <div className="w-px h-6 bg-white/[0.06] mx-2" />

            {/* Search trigger */}
            <button
              onClick={() => setSearchOpen(true)}
              className="flex items-center gap-2 px-3 py-1.5 text-xs text-zinc-500 rounded-lg border border-white/[0.06] hover:border-white/[0.12] hover:text-zinc-300 transition-all"
            >
              <Search className="w-3.5 h-3.5" />
              <span className="hidden lg:inline">Search</span>
              <kbd className="hidden lg:inline text-[10px] px-1.5 py-0.5 bg-white/[0.05] rounded font-mono ml-1">
                Ctrl K
              </kbd>
            </button>

            {/* Connection status */}
            <div className={`flex items-center gap-1.5 px-2 py-1.5 text-xs ${connectionColor}`} title={`Status: ${connectionStatus}${lastFetched ? ` • Last sync: ${getTimeSince(lastFetched)}` : ''}`}>
              <ConnectionIcon className="w-3.5 h-3.5" />
              <span className="hidden lg:inline text-[10px] capitalize">{connectionStatus}</span>
            </div>

            {/* Sound toggle */}
            <button
              onClick={() => setSoundEnabled(!soundEnabled)}
              className={`flex items-center gap-1.5 px-2 py-1.5 text-xs rounded-lg transition-colors ${
                soundEnabled 
                  ? "text-radar-green hover:bg-radar-green/10" 
                  : "text-zinc-500 hover:text-zinc-300 hover:bg-white/[0.04]"
              }`}
              title={soundEnabled ? "Mute critical alert sounds" : "Enable critical alert sounds"}
            >
              {soundEnabled ? <Volume2 className="w-3.5 h-3.5" /> : <VolumeX className="w-3.5 h-3.5" />}
            </button>

            {/* Watchlist badge */}
            {watchlistCount > 0 && (
              <Link
                href="/species?watched=true"
                className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs rounded-lg text-red-400 hover:bg-red-500/10 transition-colors"
              >
                <Heart className="w-3.5 h-3.5 fill-current" />
                <span className="font-mono">{watchlistCount}</span>
              </Link>
            )}

            {/* User menu */}
            {session ? (
              <div className="relative" ref={userMenuRef}>
                <button
                  onClick={() => setUserMenuOpen(!userMenuOpen)}
                  className="flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-white/[0.04] transition-colors ml-1"
                >
                  {session.user?.image ? (
                    <img
                      src={session.user.image}
                      alt={session.user.name || "User"}
                      className="w-6 h-6 rounded-full border border-white/[0.1]"
                    />
                  ) : (
                    <div className="w-6 h-6 rounded-full bg-radar-green/20 border border-radar-green/30 flex items-center justify-center">
                      <User className="w-3 h-3 text-radar-green" />
                    </div>
                  )}
                  <ChevronDown className={`w-3 h-3 text-zinc-500 transition-transform ${userMenuOpen ? "rotate-180" : ""}`} />
                </button>

                <AnimatePresence>
                  {userMenuOpen && (
                    <motion.div
                      initial={{ opacity: 0, y: -4, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: -4, scale: 0.95 }}
                      transition={{ duration: 0.15 }}
                      className="absolute right-0 top-full mt-2 w-56 rounded-xl bg-[#0c0c0e] border border-white/[0.08] shadow-2xl overflow-hidden z-50"
                    >
                      <div className="px-4 py-3 border-b border-white/[0.06]">
                        <p className="text-sm font-medium text-white truncate">
                          {session.user?.name || "User"}
                        </p>
                        <p className="text-xs text-zinc-500 truncate">
                          {session.user?.email}
                        </p>
                      </div>
                      <div className="p-1.5">
                        <button
                          onClick={() => {
                            setUserMenuOpen(false);
                            signOut({ callbackUrl: "/" });
                          }}
                          className="w-full flex items-center gap-2 px-3 py-2 text-sm text-zinc-400 rounded-lg hover:text-white hover:bg-white/[0.04] transition-colors"
                        >
                          <LogOut className="w-4 h-4" />
                          Sign out
                        </button>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            ) : (
              <button
                onClick={() => signIn("github")}
                className="flex items-center gap-2 px-3 py-1.5 text-xs rounded-lg bg-radar-green/10 text-radar-green border border-radar-green/20 hover:bg-radar-green/20 transition-colors ml-1"
              >
                <LogIn className="w-3.5 h-3.5" />
                Sign in
              </button>
            )}
          </div>

          {/* Mobile: search + toggle */}
          <div className="flex md:hidden items-center gap-2">
            <button
              onClick={() => setSearchOpen(true)}
              className="p-2 text-zinc-400 hover:text-white"
            >
              <Search className="w-5 h-5" />
            </button>
            <button
              onClick={() => setMobileOpen(!mobileOpen)}
              className="p-2 text-zinc-400 hover:text-white"
            >
              {mobileOpen ? (
                <X className="w-5 h-5" />
              ) : (
                <Menu className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile menu */}
        <AnimatePresence>
          {mobileOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden bg-[#09090b]/95 backdrop-blur-xl border-b border-radar-green/10"
            >
              <div className="px-6 py-4 flex flex-col gap-2">
                {navLinks.map((link) => {
                  const active = isActive(link.href);
                  const Icon = link.icon;
                  return (
                    <Link
                      key={link.href}
                      href={link.href}
                      onClick={() => setMobileOpen(false)}
                      className={`flex items-center justify-between px-4 py-3 text-sm rounded-lg transition-colors ${
                        active
                          ? "text-radar-green bg-radar-green/10 border border-radar-green/20"
                          : "text-zinc-400 hover:text-white hover:bg-white/[0.04]"
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <Icon className="w-4 h-4" />
                        {link.label}
                      </div>
                      {link.href === "/alerts" && criticalCount > 0 && (
                        <span className="px-2 py-0.5 text-xs bg-red-500/20 text-red-400 rounded-full">
                          {criticalCount}
                        </span>
                      )}
                    </Link>
                  );
                })}
                
                {/* Mobile controls */}
                <div className="flex items-center gap-2 px-4 py-2">
                  <button
                    onClick={() => setSoundEnabled(!soundEnabled)}
                    className={`flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-colors ${
                      soundEnabled 
                        ? "text-radar-green" 
                        : "text-zinc-500"
                    }`}
                  >
                    {soundEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
                    {soundEnabled ? 'Sound On' : 'Sound Off'}
                  </button>
                  
                  <div className={`flex items-center gap-1.5 px-3 py-2 text-sm ${connectionColor}`}>
                    <ConnectionIcon className="w-4 h-4" />
                    <span className="capitalize">{connectionStatus}</span>
                  </div>
                </div>
                
                {watchlistCount > 0 && (
                  <Link
                    href="/species?watched=true"
                    onClick={() => setMobileOpen(false)}
                    className="flex items-center gap-2 px-4 py-3 text-sm rounded-lg text-red-400 hover:bg-red-500/10 transition-colors"
                  >
                    <Heart className="w-4 h-4 fill-current" />
                    {watchlistCount} Watched
                  </Link>
                )}

                {/* Mobile auth */}
                <div className="border-t border-white/[0.06] pt-2 mt-1">
                  {session ? (
                    <div className="flex items-center justify-between px-4 py-2">
                      <div className="flex items-center gap-2">
                        {session.user?.image ? (
                          <img
                            src={session.user.image}
                            alt={session.user.name || "User"}
                            className="w-7 h-7 rounded-full border border-white/[0.1]"
                          />
                        ) : (
                          <div className="w-7 h-7 rounded-full bg-radar-green/20 border border-radar-green/30 flex items-center justify-center">
                            <User className="w-3.5 h-3.5 text-radar-green" />
                          </div>
                        )}
                        <span className="text-sm text-zinc-300 truncate max-w-[140px]">
                          {session.user?.name || "User"}
                        </span>
                      </div>
                      <button
                        onClick={() => {
                          setMobileOpen(false);
                          signOut({ callbackUrl: "/" });
                        }}
                        className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-zinc-400 rounded-lg hover:text-white hover:bg-white/[0.04] transition-colors"
                      >
                        <LogOut className="w-3.5 h-3.5" />
                        Sign out
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => {
                        setMobileOpen(false);
                        signIn("github");
                      }}
                      className="flex items-center gap-2 px-4 py-3 text-sm rounded-lg text-radar-green hover:bg-radar-green/10 transition-colors w-full"
                    >
                      <LogIn className="w-4 h-4" />
                      Sign in with GitHub
                    </button>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.nav>

      {/* Search modal */}
      <SearchCommand
        isOpen={searchOpen}
        onClose={() => setSearchOpen(false)}
      />
    </>
  );
}
