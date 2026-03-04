"use client";

import { signIn } from "next-auth/react";
import { motion } from "framer-motion";
import { Shield, Github } from "lucide-react";
import Link from "next/link";

export default function SignInPage() {
  return (
    <div className="min-h-screen bg-[#09090b] flex items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-sm"
      >
        {/* Logo */}
        <Link href="/" className="flex items-center justify-center gap-2 mb-10 group">
          <div className="w-10 h-10 rounded-xl bg-radar-greenDim flex items-center justify-center border border-radar-green/30 group-hover:shadow-lg group-hover:shadow-radar-green/20 transition-shadow">
            <Shield className="w-5 h-5 text-radar-green" />
          </div>
          <span className="font-semibold text-white text-lg tracking-tight">
            sentinel
          </span>
        </Link>

        {/* Card */}
        <div className="radar-card rounded-2xl p-8 border border-white/[0.06]">
          <h1 className="text-xl font-bold text-white text-center mb-1">
            Welcome back
          </h1>
          <p className="text-sm text-zinc-500 text-center mb-8">
            Sign in to access the monitoring dashboard
          </p>

          {/* GitHub button */}
          <button
            onClick={() => signIn("github", { callbackUrl: "/dashboard" })}
            className="w-full flex items-center justify-center gap-3 px-4 py-3 rounded-xl bg-white text-black font-medium text-sm hover:bg-zinc-200 transition-colors"
          >
            <Github className="w-5 h-5" />
            Continue with GitHub
          </button>

          <div className="relative my-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-white/[0.06]" />
            </div>
            <div className="relative flex justify-center">
              <span className="bg-[#0c0c0e] px-3 text-xs text-zinc-600">
                secure authentication
              </span>
            </div>
          </div>

          <p className="text-[11px] text-zinc-600 text-center leading-relaxed">
            By signing in, you agree to our monitoring terms.
            Your GitHub profile is used only for authentication.
          </p>
        </div>

        {/* Back link */}
        <p className="text-center mt-6">
          <Link
            href="/"
            className="text-xs text-zinc-500 hover:text-radar-green transition-colors"
          >
            Back to homepage
          </Link>
        </p>
      </motion.div>
    </div>
  );
}
