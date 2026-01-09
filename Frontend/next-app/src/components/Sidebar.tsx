"use client";

import Image from "next/image";
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

  const iconStyle = { width: "20px", height: "20px", strokeWidth: "2" };

  return (
    <aside className="sidebar">
      <Link href="/" className="logo" style={{ display: "flex", alignItems: "center", gap: "0.75rem", padding: "1.5rem 1.5rem 2rem", textDecoration: "none" }}>
        <Image
          src="/jett-rag-logo.png"
          alt="Jett-RAG"
          width={40}
          height={40}
          style={{ height: "auto", width: "auto", maxHeight: "40px" }}
          priority
        />
        <span style={{ fontSize: "1.25rem", fontWeight: "700", color: "white", letterSpacing: "-0.025em" }}>
          Jett-RAG
        </span>
      </Link>

      <nav className="nav-menu">
        <Link href="/dashboard" className={isActive("/dashboard")}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" className="icon" style={iconStyle}>
            <rect x="3" y="3" width="7" height="7"></rect>
            <rect x="14" y="3" width="7" height="7"></rect>
            <rect x="14" y="14" width="7" height="7"></rect>
            <rect x="3" y="14" width="7" height="7"></rect>
          </svg>
          Dashboard
        </Link>
        <Link href="/chat" className={isActive("/chat")}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" className="icon" style={iconStyle}>
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
          Chat
        </Link>
        <Link href="/memory" className={isActive("/memory")}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" className="icon" style={iconStyle}>
            <path d="M12 2a4 4 0 0 0-4 4 4 4 0 0 0 4 4 4 4 0 0 0 4-4 4 4 0 0 0-4-4"></path>
            <path d="M20 2a4 4 0 0 0-4 4 4 4 0 0 0 4 4 4 4 0 0 0 4-4 4 4 0 0 0-4-4"></path>
            <path d="M4 2a4 4 0 0 0-4 4 4 4 0 0 0 4 4 4 4 0 0 0 4-4 4 4 0 0 0-4-4"></path>
            <path d="M12 14v10"></path>
            <path d="M4 14v10"></path>
            <path d="M20 14v10"></path>
          </svg>
          Memory
        </Link>
        <Link href="/settings" className={isActive("/settings")}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" className="icon" style={iconStyle}>
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
          </svg>
          Settings
        </Link>

        {/* Admin-only navigation */}
        {isAdmin && (
          <>
            <div className="nav-divider" style={{ margin: "1.5rem 0 0.5rem", padding: "0 1rem", fontSize: "0.75rem", textTransform: "uppercase", color: "#6b7280", fontWeight: "600", letterSpacing: "0.05em" }}>
              Admin
            </div>
            <Link href="/analytics" className={isActive("/analytics")}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" className="icon" style={iconStyle}>
                <line x1="18" y1="20" x2="18" y2="10"></line>
                <line x1="12" y1="20" x2="12" y2="4"></line>
                <line x1="6" y1="20" x2="6" y2="14"></line>
              </svg>
              Analytics
            </Link>
            <Link href="/admin" className={isActive("/admin")}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" className="icon" style={iconStyle}>
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                <circle cx="9" cy="7" r="4"></circle>
                <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
              </svg>
              User Management
            </Link>
            <Link href="/documents" className={isActive("/documents")}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" className="icon" style={iconStyle}>
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                <polyline points="14 2 14 8 20 8"></polyline>
                <line x1="16" y1="13" x2="8" y2="13"></line>
                <line x1="16" y1="17" x2="8" y2="17"></line>
                <polyline points="10 9 9 9 8 9"></polyline>
              </svg>
              Documents
            </Link>
            <Link href="/admin/graph" className={isActive("/admin/graph")}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" className="icon" style={iconStyle}>
                <circle cx="18" cy="5" r="3"></circle>
                <circle cx="6" cy="12" r="3"></circle>
                <circle cx="18" cy="19" r="3"></circle>
                <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
                <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
              </svg>
              Knowledge Graph
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
        <button onClick={handleLogout} className="nav-item logout-link" style={{ width: "100%", justifyContent: "flex-start", marginTop: "0.5rem" }}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" className="icon" style={iconStyle}>
            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
            <polyline points="16 17 21 12 16 7"></polyline>
            <line x1="21" y1="12" x2="9" y2="12"></line>
          </svg>
          Logout
        </button>
      </div>
    </aside>
  );
}
