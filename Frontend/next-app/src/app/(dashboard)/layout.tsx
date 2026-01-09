"use client";

import Sidebar from "@/components/Sidebar";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const [isReady, setIsReady] = useState(false);

  // Add is-ready class after mount to trigger fade-in animation
  useEffect(() => {
    setIsReady(true);
  }, []);

  return (
    <div className={`app-layout ${isReady ? "is-ready" : ""}`}>
      <Sidebar />
      <div className="dashboard-content">
        <div style={{ flex: 1, overflow: "auto" }}>{children}</div>
      </div>
    </div>
  );
}