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
    <main className="main-content settings-page">
      <header>
        <h2>Settings</h2>
        <p>Manage your account and preferences.</p>
      </header>

      {/* Profile Section */}
      <section className="settings-section">
        <h3>Profile Information</h3>
        <p className="section-description">Update your personal information.</p>

        <div className="settings-form">
          <div className="form-group">
            <label>Email</label>
            <input
              type="email"
              value={user?.email || ""}
              disabled
              className="form-input disabled"
            />
            <small>Email cannot be changed</small>
          </div>

          <div className="form-group">
            <label>Full Name</label>
            <input
              type="text"
              value={profile.name}
              onChange={(e) => setProfile({ ...profile, name: e.target.value })}
              placeholder="Enter your name"
              className="form-input"
            />
          </div>

          <div className="form-group">
            <label>Department</label>
            <input
              type="text"
              value={profile.department}
              onChange={(e) =>
                setProfile({ ...profile, department: e.target.value })
              }
              placeholder="e.g., Engineering, Marketing"
              className="form-input"
            />
          </div>

          <button
            onClick={handleSaveProfile}
            disabled={isLoading}
            className="btn btn-dark"
          >
            {isLoading ? "Saving..." : "Save Profile"}
          </button>
        </div>
      </section>

      {/* Preferences Section */}
      <section className="settings-section">
        <h3>Chat Preferences</h3>
        <p className="section-description">Customize your chat experience.</p>

        <div className="settings-options">
          <div className="setting-item">
            <div className="setting-info">
              <label>Enable Streaming</label>
              <small>Show AI responses as they are generated</small>
            </div>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={preferences.streamingEnabled}
                onChange={(e) =>
                  handlePreferenceChange("streamingEnabled", e.target.checked)
                }
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="setting-item">
            <div className="setting-info">
              <label>Notifications</label>
              <small>Receive notifications for important updates</small>
            </div>
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={preferences.notifications}
                onChange={(e) =>
                  handlePreferenceChange("notifications", e.target.checked)
                }
              />
              <span className="toggle-slider"></span>
            </label>
          </div>

          <div className="setting-item">
            <div className="setting-info">
              <label>Response Temperature</label>
              <small>Controls creativity (0 = precise, 1 = creative)</small>
            </div>
            <div className="slider-container">
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={preferences.defaultTemperature}
                onChange={(e) =>
                  handlePreferenceChange(
                    "defaultTemperature",
                    parseFloat(e.target.value)
                  )
                }
                className="range-slider"
              />
              <span className="slider-value">
                {preferences.defaultTemperature}
              </span>
            </div>
          </div>

          <div className="setting-item">
            <div className="setting-info">
              <label>Max Response Length</label>
              <small>Maximum tokens in AI responses</small>
            </div>
            <select
              value={preferences.maxTokens}
              onChange={(e) =>
                handlePreferenceChange("maxTokens", parseInt(e.target.value))
              }
              className="form-select"
            >
              <option value={512}>Short (512)</option>
              <option value={1024}>Medium (1024)</option>
              <option value={2048}>Long (2048)</option>
              <option value={4096}>Extended (4096)</option>
            </select>
          </div>
        </div>
      </section>

      {/* Theme Section */}
      <section className="settings-section">
        <h3>Appearance</h3>
        <p className="section-description">Choose your preferred theme.</p>

        <div className="theme-options">
          {(["light", "dark", "system"] as const).map((theme) => (
            <button
              key={theme}
              onClick={() => handlePreferenceChange("theme", theme)}
              className={`theme-button ${
                preferences.theme === theme ? "active" : ""
              }`}
            >
              <span className="theme-icon">
                {theme === "light" ? "☀️" : theme === "dark" ? "🌙" : "💻"}
              </span>
              <span className="theme-label">
                {theme.charAt(0).toUpperCase() + theme.slice(1)}
              </span>
            </button>
          ))}
        </div>
      </section>

      {/* Account Info */}
      <section className="settings-section account-info">
        <h3>Account Information</h3>
        <div className="info-grid">
          <div className="info-item">
            <label>Account Type</label>
            <span className={`badge ${user?.role}`}>
              {user?.role || "user"}
            </span>
          </div>
          <div className="info-item">
            <label>Member Since</label>
            <span>
              {user?.created_at
                ? new Date(user.created_at).toLocaleDateString()
                : "—"}
            </span>
          </div>
          <div className="info-item">
            <label>Last Login</label>
            <span>
              {user?.last_login_at
                ? new Date(user.last_login_at).toLocaleDateString()
                : "—"}
            </span>
          </div>
        </div>
      </section>
    </main>
  );
}
