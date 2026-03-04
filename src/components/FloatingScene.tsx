"use client";

import { useEffect, useRef, useCallback } from "react";

interface Node {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  opacity: number;
  hue: number; // 160-180 range for teal/emerald
  pulseOffset: number;
  hasAura: boolean;
}

/**
 * Canvas-based floating constellation background.
 * Renders drifting nodes with pulsing auras connected by faint lines.
 * Wildlife-themed: teal/emerald palette, organic drift patterns.
 */
export default function FloatingScene() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<Node[]>([]);
  const animRef = useRef<number>(0);
  const mouseRef = useRef({ x: -1000, y: -1000 });

  const initNodes = useCallback((w: number, h: number) => {
    const count = Math.max(18, Math.floor((w * h) / 40000));
    const nodes: Node[] = [];
    for (let i = 0; i < count; i++) {
      nodes.push({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        radius: 1.5 + Math.random() * 2.5,
        opacity: 0.15 + Math.random() * 0.35,
        hue: 160 + Math.random() * 20, // teal range
        pulseOffset: Math.random() * Math.PI * 2,
        hasAura: Math.random() < 0.25, // 25% of nodes get a glow aura
      });
    }
    nodesRef.current = nodes;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
      ctx.scale(dpr, dpr);
      initNodes(window.innerWidth, window.innerHeight);
    };

    resize();
    window.addEventListener("resize", resize);

    const handleMouse = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    };
    window.addEventListener("mousemove", handleMouse);

    const maxLinkDist = 180;
    const mouseInfluenceRadius = 200;

    const draw = (time: number) => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      ctx.clearRect(0, 0, w, h);

      const nodes = nodesRef.current;
      const t = time * 0.001;

      // Update positions
      for (const node of nodes) {
        node.x += node.vx;
        node.y += node.vy;

        // Soft bounce
        if (node.x < 0 || node.x > w) node.vx *= -1;
        if (node.y < 0 || node.y > h) node.vy *= -1;

        // Mouse repulsion (subtle)
        const dx = node.x - mouseRef.current.x;
        const dy = node.y - mouseRef.current.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < mouseInfluenceRadius && dist > 0) {
          const force = (mouseInfluenceRadius - dist) / mouseInfluenceRadius * 0.015;
          node.vx += (dx / dist) * force;
          node.vy += (dy / dist) * force;
        }

        // Clamp velocity
        const speed = Math.sqrt(node.vx * node.vx + node.vy * node.vy);
        if (speed > 0.5) {
          node.vx = (node.vx / speed) * 0.5;
          node.vy = (node.vy / speed) * 0.5;
        }
      }

      // Draw connecting lines
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i];
          const b = nodes[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < maxLinkDist) {
            const alpha = (1 - dist / maxLinkDist) * 0.06;
            const pulse = Math.sin(t * 0.5 + i * 0.3) * 0.02 + alpha;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = `hsla(170, 60%, 60%, ${Math.max(0, pulse)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      // Draw nodes
      for (const node of nodes) {
        const pulse = Math.sin(t * 1.2 + node.pulseOffset) * 0.15 + node.opacity;

        // Aura glow for selected nodes
        if (node.hasAura) {
          const auraSize = node.radius * 12 + Math.sin(t * 0.8 + node.pulseOffset) * 6;
          const gradient = ctx.createRadialGradient(
            node.x, node.y, 0,
            node.x, node.y, auraSize
          );
          gradient.addColorStop(0, `hsla(${node.hue}, 60%, 60%, 0.08)`);
          gradient.addColorStop(0.5, `hsla(${node.hue}, 60%, 60%, 0.03)`);
          gradient.addColorStop(1, "transparent");
          ctx.beginPath();
          ctx.arc(node.x, node.y, auraSize, 0, Math.PI * 2);
          ctx.fillStyle = gradient;
          ctx.fill();
        }

        // Core dot
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = `hsla(${node.hue}, 60%, 65%, ${Math.max(0.1, pulse)})`;
        ctx.fill();

        // Outer ring for larger nodes
        if (node.radius > 3) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.radius + 3, 0, Math.PI * 2);
          ctx.strokeStyle = `hsla(${node.hue}, 50%, 60%, ${Math.max(0, pulse * 0.3)})`;
          ctx.lineWidth = 0.5;
          ctx.stroke();
        }
      }

      animRef.current = requestAnimationFrame(draw);
    };

    animRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(animRef.current);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", handleMouse);
    };
  }, [initNodes]);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none z-0"
      style={{ opacity: 0.7 }}
    />
  );
}
