"use client";

import { useAuth } from "@/contexts/AuthContext";
import Link from "next/link";

export default function Dashboard() {
  const { user, isAuthenticated } = useAuth();

  const isAdmin = user?.role === "admin";

  return (
    <main className="main-content">
      <header>
        <h2>Dashboard</h2>
        <p>
          Welcome{user?.full_name ? `, ${user.full_name}` : ""} to your GraphRAG
          Knowledge Portal.
        </p>
      </header>

      <div
        className="dashboard-grid"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: "1.5rem",
        }}
      >
        <Link
          href="/chat"
          className="panel"
          style={{ textDecoration: "none", color: "inherit" }}
        >
          <h3>💬 Chat</h3>
          <p>Ask questions about your documents using AI-powered search.</p>
        </Link>

        <Link
          href="/settings"
          className="panel"
          style={{ textDecoration: "none", color: "inherit" }}
        >
          <h3>⚙️ Settings</h3>
          <p>Manage your profile and customize your preferences.</p>
        </Link>

        {/* Admin-only sections */}
        {isAdmin && (
          <>
            <Link
              href="/analytics"
              className="panel admin-panel"
              style={{ textDecoration: "none", color: "inherit" }}
            >
              <h3>📊 Analytics</h3>
              <p>View system performance metrics and usage statistics.</p>
              <span className="admin-badge">Admin</span>
            </Link>

            <Link
              href="/admin"
              className="panel admin-panel"
              style={{ textDecoration: "none", color: "inherit" }}
            >
              <h3>👥 User Management</h3>
              <p>Manage users, roles, and document permissions.</p>
              <span className="admin-badge">Admin</span>
            </Link>

            <Link
              href="/documents"
              className="panel admin-panel"
              style={{ textDecoration: "none", color: "inherit" }}
            >
              <h3>📁 Documents</h3>
              <p>Upload, manage, and monitor document indexing.</p>
              <span className="admin-badge">Admin</span>
            </Link>
          </>
        )}
      </div>

      {!isAuthenticated && (
        <div
          className="panel"
          style={{
            marginTop: "1.5rem",
            backgroundColor: "#fef3c7",
            borderColor: "#f59e0b",
          }}
        >
          <p style={{ color: "#92400e" }}>
            You are not signed in.{" "}
            <Link href="/sign-in" style={{ color: "#d97706", fontWeight: 500 }}>
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
            marginTop: "1.5rem",
            backgroundColor: "#eff6ff",
            borderColor: "#3b82f6",
          }}
        >
          <p style={{ color: "#1e40af" }}>
            💡 <strong>Tip:</strong> Use the Chat feature to ask questions about
            your organization&apos;s documents. The AI will find relevant
            information and cite its sources.
          </p>
        </div>
      )}
    </main>
  );
}
