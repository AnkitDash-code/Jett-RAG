"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { api } from "@/lib/api";
import { toast } from "sonner";

interface UserPreferences {
  theme: "light" | "dark" | "system";
  notifications: boolean;
  streamingEnabled: boolean;
  defaultTemperature: number;
  maxTokens: number;
}

export default function Settings() {
  const { user } = useAuth();
  const [isLoading, setIsLoading] = useState(false);
  const [preferences, setPreferences] = useState<UserPreferences>({
    theme: "system",
    notifications: true,
    streamingEnabled: true,
    defaultTemperature: 0.7,
    maxTokens: 1024,
  });
  const [profile, setProfile] = useState({
    name: "",
    department: "",
  });

  useEffect(() => {
    if (user) {
      setProfile({
        name: user.full_name || "",
        department: user.department || "",
      });
    }
  }, [user]);

  const handleSaveProfile = async () => {
    try {
      setIsLoading(true);
      await api.updateCurrentUser({
        name: profile.name,
        department: profile.department,
      });
      toast.success("Profile updated successfully");
    } catch (err) {
      toast.error((err as Error).message || "Failed to update profile");
    } finally {
      setIsLoading(false);
    }
  };

  const handlePreferenceChange = (
    key: keyof UserPreferences,
    value: unknown
  ) => {
    setPreferences((prev) => ({ ...prev, [key]: value }));
    // Save preferences to localStorage
    const updated = { ...preferences, [key]: value };
    localStorage.setItem("user_preferences", JSON.stringify(updated));
    toast.success("Preference saved");
  };

  return (
    <main className="main-content">
      <header style={{ marginBottom: "2rem" }}>
        <h2 style={{ fontSize: "2rem", marginBottom: "0.5rem", background: "linear-gradient(to right, #fff, #9ca3af)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>Settings</h2>
        <p style={{ color: "#9ca3af" }}>
          Manage your account and preferences.
        </p>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))", gap: "1.5rem" }}>

        {/* Left Column */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          {/* Profile Section */}
          <section className="feature-card-modern" style={{ background: "#0a0a0a", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem", padding: "1.5rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1.5rem" }}>
              <span style={{ padding: "0.5rem", borderRadius: "0.5rem", background: "rgba(59, 130, 246, 0.1)", color: "#3b82f6" }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
              </span>
              <div>
                <h3 style={{ fontSize: "1.125rem", fontWeight: "600", color: "white" }}>Profile Information</h3>
                <p style={{ color: "#9ca3af", fontSize: "0.875rem" }}>Update your personal details</p>
              </div>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "1.25rem" }}>
              <div>
                <label style={{ display: "block", color: "#d1d5db", fontSize: "0.875rem", marginBottom: "0.5rem" }}>Email Address</label>
                <div style={{ position: "relative" }}>
                  <input
                    type="email"
                    value={user?.email || ""}
                    disabled
                    style={{ width: "100%", padding: "0.75rem", paddingLeft: "2.5rem", background: "rgba(255, 255, 255, 0.03)", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "0.5rem", color: "#6b7280", cursor: "not-allowed" }}
                  />
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ position: "absolute", left: "0.75rem", top: "50%", transform: "translateY(-50%)" }}><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path><polyline points="22,6 12,13 2,6"></polyline></svg>
                </div>
              </div>

              <div>
                <label style={{ display: "block", color: "#d1d5db", fontSize: "0.875rem", marginBottom: "0.5rem" }}>Full Name</label>
                <input
                  type="text"
                  value={profile.name}
                  onChange={(e) => setProfile({ ...profile, name: e.target.value })}
                  placeholder="Enter your name"
                  style={{ width: "100%", padding: "0.75rem", background: "#1f2937", border: "1px solid #374151", borderRadius: "0.5rem", color: "white", outline: "none" }}
                  onFocus={(e) => e.target.style.borderColor = "#3b82f6"}
                  onBlur={(e) => e.target.style.borderColor = "#374151"}
                />
              </div>

              <div>
                <label style={{ display: "block", color: "#d1d5db", fontSize: "0.875rem", marginBottom: "0.5rem" }}>Department</label>
                <input
                  type="text"
                  value={profile.department}
                  onChange={(e) => setProfile({ ...profile, department: e.target.value })}
                  placeholder="e.g., Engineering"
                  style={{ width: "100%", padding: "0.75rem", background: "#1f2937", border: "1px solid #374151", borderRadius: "0.5rem", color: "white", outline: "none" }}
                  onFocus={(e) => e.target.style.borderColor = "#3b82f6"}
                  onBlur={(e) => e.target.style.borderColor = "#374151"}
                />
              </div>

              <button
                onClick={handleSaveProfile}
                disabled={isLoading}
                style={{
                  marginTop: "0.5rem",
                  width: "100%",
                  padding: "0.75rem",
                  borderRadius: "0.5rem",
                  background: "#3b82f6",
                  color: "white",
                  border: "none",
                  fontWeight: "600",
                  cursor: isLoading ? "not-allowed" : "pointer",
                  opacity: isLoading ? 0.7 : 1,
                  transition: "background 0.2s"
                }}
              >
                {isLoading ? "Saving..." : "Save Changes"}
              </button>
            </div>
          </section>

          {/* Account Info */}
          <section className="feature-card-modern" style={{ background: "#0a0a0a", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem", padding: "1.5rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1.5rem" }}>
              <span style={{ padding: "0.5rem", borderRadius: "0.5rem", background: "rgba(16, 185, 129, 0.1)", color: "#10b981" }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
              </span>
              <div>
                <h3 style={{ fontSize: "1.125rem", fontWeight: "600", color: "white" }}>Account Status</h3>
                <p style={{ color: "#9ca3af", fontSize: "0.875rem" }}>Your subscription and role details</p>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "1rem" }}>
              <div style={{ padding: "1rem", background: "rgba(255, 255, 255, 0.03)", borderRadius: "0.5rem", border: "1px solid rgba(255, 255, 255, 0.05)" }}>
                <span style={{ color: "#9ca3af", fontSize: "0.75rem", display: "block", marginBottom: "0.25rem" }}>Role</span>
                <span style={{ color: "white", fontWeight: "600", textTransform: "capitalize" }}>{user?.role || "user"}</span>
              </div>
              <div style={{ padding: "1rem", background: "rgba(255, 255, 255, 0.03)", borderRadius: "0.5rem", border: "1px solid rgba(255, 255, 255, 0.05)" }}>
                <span style={{ color: "#9ca3af", fontSize: "0.75rem", display: "block", marginBottom: "0.25rem" }}>Member Since</span>
                <span style={{ color: "white", fontWeight: "600" }}>{user?.created_at ? new Date(user.created_at).toLocaleDateString() : "—"}</span>
              </div>
            </div>
          </section>
        </div>

        {/* Right Column */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          {/* Chat Preferences */}
          <section className="feature-card-modern" style={{ background: "#0a0a0a", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem", padding: "1.5rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "1.5rem" }}>
              <span style={{ padding: "0.5rem", borderRadius: "0.5rem", background: "rgba(245, 158, 11, 0.1)", color: "#f59e0b" }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
              </span>
              <div>
                <h3 style={{ fontSize: "1.125rem", fontWeight: "600", color: "white" }}>Chat Preferences</h3>
                <p style={{ color: "#9ca3af", fontSize: "0.875rem" }}>Customize model behavior</p>
              </div>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <label style={{ display: "block", color: "white", fontWeight: "500" }}>Enable Streaming</label>
                  <small style={{ color: "#9ca3af" }}>Show AI responses as they are generated</small>
                </div>
                <input
                  type="checkbox"
                  checked={preferences.streamingEnabled}
                  onChange={(e) => handlePreferenceChange("streamingEnabled", e.target.checked)}
                  style={{ accentColor: "#3b82f6", width: "1.25rem", height: "1.25rem" }}
                />
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <label style={{ display: "block", color: "white", fontWeight: "500" }}>Notifications</label>
                  <small style={{ color: "#9ca3af" }}>Receive alerts for updates</small>
                </div>
                <input
                  type="checkbox"
                  checked={preferences.notifications}
                  onChange={(e) => handlePreferenceChange("notifications", e.target.checked)}
                  style={{ accentColor: "#3b82f6", width: "1.25rem", height: "1.25rem" }}
                />
              </div>

              <div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                  <label style={{ color: "#d1d5db", fontSize: "0.875rem" }}>Creativity (Temperature)</label>
                  <span style={{ color: "#3b82f6", fontWeight: "600", fontSize: "0.875rem" }}>{preferences.defaultTemperature}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={preferences.defaultTemperature}
                  onChange={(e) => handlePreferenceChange("defaultTemperature", parseFloat(e.target.value))}
                  style={{ width: "100%", accentColor: "#3b82f6" }}
                />
                <div style={{ display: "flex", justifyContent: "space-between", marginTop: "0.25rem" }}>
                  <small style={{ color: "#6b7280", fontSize: "0.75rem" }}>Precise</small>
                  <small style={{ color: "#6b7280", fontSize: "0.75rem" }}>Creative</small>
                </div>
              </div>

              <div>
                <label style={{ display: "block", color: "#d1d5db", fontSize: "0.875rem", marginBottom: "0.5rem" }}>Max Response Length</label>
                <select
                  value={preferences.maxTokens}
                  onChange={(e) => handlePreferenceChange("maxTokens", parseInt(e.target.value))}
                  style={{ width: "100%", padding: "0.75rem", background: "#1f2937", border: "1px solid #374151", borderRadius: "0.5rem", color: "white", outline: "none" }}
                >
                  <option value={512}>Short (512 tokens)</option>
                  <option value={1024}>Medium (1024 tokens)</option>
                  <option value={2048}>Long (2048 tokens)</option>
                  <option value={4096}>Extended (4096 tokens)</option>
                </select>
              </div>
            </div>
          </section>


        </div>

      </div>
    </main>
  );
}
