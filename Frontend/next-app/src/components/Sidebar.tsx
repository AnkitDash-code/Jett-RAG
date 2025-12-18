"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/contexts/AuthContext";

export default function Sidebar() {
  const pathname = usePathname();
  const { user, logout } = useAuth();

  const isAdmin = user?.role === "admin";

  const isActive = (path: string) => {
    return pathname.startsWith(path) ? "nav-item active" : "nav-item";
  };

  const handleLogout = async () => {
    await logout();
  };

  return (
    <aside className="sidebar">
      <div className="logo">
        <span className="text-blue-500 text-2xl">G</span> GraphRAG
      </div>
      <nav className="nav-menu">
        <Link href="/dashboard" className={isActive("/dashboard")}>
          <i className="icon">🏠</i> Dashboard
        </Link>
        <Link href="/chat" className={isActive("/chat")}>
          <i className="icon">💬</i> Chat
        </Link>
        <Link href="/settings" className={isActive("/settings")}>
          <i className="icon">⚙️</i> Settings
        </Link>

        {/* Admin-only navigation */}
        {isAdmin && (
          <>
            <div className="nav-divider">
              <span>Admin</span>
            </div>
            <Link href="/analytics" className={isActive("/analytics")}>
              <i className="icon">📊</i> Analytics
            </Link>
            <Link href="/admin" className={isActive("/admin")}>
              <i className="icon">👥</i> User Management
            </Link>
            <Link href="/documents" className={isActive("/documents")}>
              <i className="icon">📁</i> Documents
            </Link>
          </>
        )}
      </nav>
      <div className="logout">
        <div className="user-info">
          <span className="user-name">
            {user?.full_name || user?.email?.split("@")[0] || "User"}
          </span>
          <span className={`user-role ${isAdmin ? "admin" : "user"}`}>
            {isAdmin ? "Admin" : "User"}
          </span>
        </div>
        <button onClick={handleLogout} className="nav-item logout-link">
          <i className="icon">🚪</i> Logout
        </button>
      </div>
    </aside>
  );
}
