"use client";

import { useState, useEffect, useCallback } from "react";
import { api, API_BASE_URL } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";

interface Session {
  id: string;
  device_name: string;
  device_type: string;
  ip_address: string;
  user_agent: string;
  location?: string;
  created_at: string;
  last_active: string;
  is_current: boolean;
}

export default function SessionsPage() {
  const { user } = useAuth();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const fetchSessions = useCallback(async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/sessions`, {
        headers: {
          Authorization: `Bearer ${api.getAccessToken()}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setSessions(data.sessions || []);
      } else {
        throw new Error("Failed to fetch sessions");
      }
    } catch (err) {
      console.error("Sessions fetch error:", err);
      // Demo data
      setSessions([
        {
          id: "1",
          device_name: "Chrome on Windows",
          device_type: "desktop",
          ip_address: "192.168.1.100",
          user_agent: "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0",
          location: "Local Network",
          created_at: new Date().toISOString(),
          last_active: new Date().toISOString(),
          is_current: true,
        },
        {
          id: "2",
          device_name: "Safari on iPhone",
          device_type: "mobile",
          ip_address: "192.168.1.101",
          user_agent: "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0)",
          location: "Local Network",
          created_at: new Date(Date.now() - 86400000).toISOString(),
          last_active: new Date(Date.now() - 3600000).toISOString(),
          is_current: false,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  const handleRevoke = async (sessionId: string) => {
    if (!confirm("Are you sure you want to revoke this session?")) return;

    try {
      const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${api.getAccessToken()}`,
        },
      });

      if (response.ok) {
        setSessions((prev) => prev.filter((s) => s.id !== sessionId));
        toast.success("Session revoked");
      } else {
        throw new Error("Failed to revoke session");
      }
    } catch (err) {
      toast.error("Failed to revoke session");
    }
  };

  const handleLogoutAll = async () => {
    if (!confirm("This will log you out from all devices. Are you sure?"))
      return;

    try {
      const response = await fetch(`${API_BASE_URL}/sessions/logout-all`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${api.getAccessToken()}`,
        },
      });

      if (response.ok) {
        toast.success("Logged out from all devices");
        // Redirect to login
        window.location.href = "/sign-in";
      } else {
        throw new Error("Failed to logout all");
      }
    } catch (err) {
      toast.error("Failed to logout from all devices");
    }
  };

  const getDeviceIcon = (type: string) => {
    switch (type) {
      case "mobile":
        return "📱";
      case "tablet":
        return "📱";
      case "desktop":
        return "💻";
      default:
        return "🖥️";
    }
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  return (
    <main className="main-content sessions-page">
      <header>
        <h2>Active Sessions</h2>
        <p>Manage your logged-in devices and sessions.</p>
      </header>

      {/* Actions */}
      <div
        style={{
          display: "flex",
          justifyContent: "flex-end",
          marginBottom: "1rem",
        }}
      >
        <button
          onClick={handleLogoutAll}
          style={{
            padding: "0.5rem 1rem",
            borderRadius: "0.375rem",
            backgroundColor: "#dc2626",
            color: "white",
            border: "none",
            cursor: "pointer",
            fontSize: "0.875rem",
          }}
        >
          Logout All Devices
        </button>
      </div>

      {/* Sessions List */}
      {isLoading ? (
        <div className="panel" style={{ textAlign: "center", padding: "2rem" }}>
          Loading sessions...
        </div>
      ) : sessions.length === 0 ? (
        <div className="panel" style={{ textAlign: "center", padding: "2rem" }}>
          No active sessions found.
        </div>
      ) : (
        <div
          style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}
        >
          {sessions.map((session) => (
            <div
              key={session.id}
              className="panel"
              style={{
                display: "flex",
                alignItems: "center",
                gap: "1rem",
                padding: "1rem",
                border: session.is_current
                  ? "2px solid #10b981"
                  : "1px solid #374151",
              }}
            >
              {/* Device Icon */}
              <div
                style={{
                  fontSize: "2rem",
                  width: "48px",
                  height: "48px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  backgroundColor: "#1f2937",
                  borderRadius: "0.5rem",
                }}
              >
                {getDeviceIcon(session.device_type)}
              </div>

              {/* Device Info */}
              <div style={{ flex: 1 }}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "0.5rem",
                    marginBottom: "0.25rem",
                  }}
                >
                  <h4 style={{ margin: 0, color: "#f9fafb" }}>
                    {session.device_name}
                  </h4>
                  {session.is_current && (
                    <span
                      style={{
                        backgroundColor: "#065f46",
                        color: "#10b981",
                        padding: "0.125rem 0.5rem",
                        borderRadius: "9999px",
                        fontSize: "0.625rem",
                        fontWeight: 500,
                      }}
                    >
                      Current
                    </span>
                  )}
                </div>
                <div style={{ fontSize: "0.8125rem", color: "#9ca3af" }}>
                  <span>{session.ip_address}</span>
                  {session.location && <span> • {session.location}</span>}
                </div>
                <div style={{ fontSize: "0.75rem", color: "#6b7280" }}>
                  Last active: {formatDate(session.last_active)}
                </div>
              </div>

              {/* Actions */}
              {!session.is_current && (
                <button
                  onClick={() => handleRevoke(session.id)}
                  style={{
                    padding: "0.375rem 0.75rem",
                    borderRadius: "0.375rem",
                    backgroundColor: "transparent",
                    color: "#ef4444",
                    border: "1px solid #ef4444",
                    cursor: "pointer",
                    fontSize: "0.75rem",
                  }}
                >
                  Revoke
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Security Tips */}
      <div
        className="panel"
        style={{
          marginTop: "2rem",
          backgroundColor: "#1e3a5f",
          borderColor: "#3b82f6",
        }}
      >
        <h4 style={{ margin: "0 0 0.5rem 0", color: "#60a5fa" }}>
          🔒 Security Tips
        </h4>
        <ul
          style={{
            margin: 0,
            paddingLeft: "1.25rem",
            color: "#93c5fd",
            fontSize: "0.875rem",
          }}
        >
          <li>Revoke sessions you don&apos;t recognize</li>
          <li>
            Use &quot;Logout All Devices&quot; if you suspect unauthorized
            access
          </li>
          <li>Enable two-factor authentication for added security</li>
        </ul>
      </div>
    </main>
  );
}
