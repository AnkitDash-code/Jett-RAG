"use client";

import { useAuth } from "@/contexts/AuthContext";
import Link from "next/link";
import { useRouter } from "next/navigation";

export default function Dashboard() {
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();

  const isAdmin = user?.role === "admin";

  return (
    <main className="main-content">
      <header style={{ marginBottom: "3rem" }}>
        <h2 style={{ fontSize: "2rem", marginBottom: "0.5rem", background: "linear-gradient(to right, #fff, #9ca3af)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>Dashboard</h2>
        <p style={{ color: "#9ca3af" }}>
          Welcome back{user?.full_name ? `, ${user.full_name}` : ""}
        </p>
      </header>

      <div
        className="dashboard-grid"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
          gap: "1.5rem",
        }}
      >
        <div
          onClick={() => router.push('/chat')}
          className="feature-card-modern"
          style={{ cursor: "pointer", minHeight: "200px" }}
        >
          <div style={{ marginBottom: "1.5rem", width: "48px", height: "48px", borderRadius: "12px", background: "rgba(59, 130, 246, 0.2)", display: "flex", alignItems: "center", justifyContent: "center", color: "#3b82f6" }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
            </svg>
          </div>
          <h3>Chat & Query</h3>
          <p>Interact with your knowledge base using advanced AI. Ask questions and get cited answers.</p>
        </div>

        <div
          onClick={() => router.push('/settings')}
          className="feature-card-modern"
          style={{ cursor: "pointer", minHeight: "200px" }}
        >
          <div style={{ marginBottom: "1.5rem", width: "48px", height: "48px", borderRadius: "12px", background: "rgba(139, 92, 246, 0.2)", display: "flex", alignItems: "center", justifyContent: "center", color: "#8b5cf6" }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3"></circle>
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
            </svg>
          </div>
          <h3>System Settings</h3>
          <p>Configure your workspace, manage account preferences, and customize your experience.</p>
        </div>

        {/* Admin-only sections */}
        {isAdmin && (
          <>
            <div
              onClick={() => router.push('/analytics')}
              className="feature-card-modern"
              style={{ cursor: "pointer", minHeight: "200px", borderColor: "rgba(16, 185, 129, 0.2)" }}
            >
              <div style={{ marginBottom: "1.5rem", width: "48px", height: "48px", borderRadius: "12px", background: "rgba(16, 185, 129, 0.2)", display: "flex", alignItems: "center", justifyContent: "center", color: "#10b981" }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="20" x2="18" y2="10"></line>
                  <line x1="12" y1="20" x2="12" y2="4"></line>
                  <line x1="6" y1="20" x2="6" y2="14"></line>
                </svg>
              </div>
              <h3>Analytics</h3>
              <p>Monitor system usage, query performance, and user engagement metrics.</p>
            </div>

            <div
              onClick={() => router.push('/admin')}
              className="feature-card-modern"
              style={{ cursor: "pointer", minHeight: "200px", borderColor: "rgba(245, 158, 11, 0.2)" }}
            >
              <div style={{ marginBottom: "1.5rem", width: "48px", height: "48px", borderRadius: "12px", background: "rgba(245, 158, 11, 0.2)", display: "flex", alignItems: "center", justifyContent: "center", color: "#f59e0b" }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                  <circle cx="9" cy="7" r="4"></circle>
                  <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                  <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
                </svg>
              </div>
              <h3>User Management</h3>
              <p>Control access, manage user roles, and oversee platform security.</p>
            </div>

            <div
              onClick={() => router.push('/documents')}
              className="feature-card-modern"
              style={{ cursor: "pointer", minHeight: "200px", borderColor: "rgba(236, 72, 153, 0.2)" }}
            >
              <div style={{ marginBottom: "1.5rem", width: "48px", height: "48px", borderRadius: "12px", background: "rgba(236, 72, 153, 0.2)", display: "flex", alignItems: "center", justifyContent: "center", color: "#ec4899" }}>
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                  <polyline points="14 2 14 8 20 8"></polyline>
                  <line x1="16" y1="13" x2="8" y2="13"></line>
                  <line x1="16" y1="17" x2="8" y2="17"></line>
                  <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
              </div>
              <h3>Documents</h3>
              <p>Upload new files, manage existing documents, and check indexing status.</p>
            </div>
          </>
        )}
      </div>

      {!isAuthenticated && (
        <div
          className="panel"
          style={{
            marginTop: "1.5rem",
            backgroundColor: "rgba(254, 243, 199, 0.1)",
            borderColor: "rgba(245, 158, 11, 0.5)",
          }}
        >
          <p style={{ color: "#fbbf24" }}>
            You are not signed in.{" "}
            <Link href="/sign-in" style={{ color: "#f59e0b", fontWeight: 700, textDecoration: "underline" }}>
              Sign in
            </Link>{" "}
            to access all features.
          </p>
        </div>
      )}

      {isAuthenticated && !isAdmin && (
        <div
          className="panel info-panel"
          style={{
            marginTop: "2rem",
            background: "linear-gradient(90deg, rgba(59, 130, 246, 0.1) 0%, rgba(30, 58, 138, 0.1) 100%)",
            borderColor: "rgba(59, 130, 246, 0.3)",
          }}
        >
          <div style={{ display: "flex", gap: "1rem", alignItems: "flex-start" }}>
            <span style={{ fontSize: "1.5rem" }}>💡</span>
            <div>
              <h4 style={{ color: "white", marginBottom: "0.25rem" }}>Pro Tip</h4>
              <p style={{ color: "#9ca3af", fontSize: "0.95rem" }}>
                Use the Chat feature to ask questions about your documents. The AI will find relevant information and cite its sources automatically.
              </p>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
