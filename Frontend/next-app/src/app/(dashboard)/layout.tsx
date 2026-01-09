"use client";

import Sidebar from "@/components/Sidebar";
import { useEffect, useState, useRef } from "react";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [isReady, setIsReady] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Add is-ready class after mount to trigger fade-in animation
  useEffect(() => {
    setIsReady(true);
  }, []);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    containerRef.current.style.setProperty("--mouse-x", `${x}px`);
    containerRef.current.style.setProperty("--mouse-y", `${y}px`);
  };

  return (
    <div className={`app-layout ${isReady ? "is-ready" : ""}`}>
      <Sidebar />
      <div
        className="dashboard-content"
        ref={containerRef}
        onMouseMove={handleMouseMove}
        style={
          {
            "--mouse-x": "50%",
            "--mouse-y": "50%",
          } as React.CSSProperties
        }
      >
        <div style={{ flex: 1, overflow: "auto", position: "relative", zIndex: 1 }}>
          {children}
        </div>
      </div>
    </div>
  );
}