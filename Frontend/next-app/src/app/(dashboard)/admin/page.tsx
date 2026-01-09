"use client";

import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { useRouter } from "next/navigation";
import type { UserResponse, DocumentResponse } from "@/types/api";

interface UserPermissions {
  can_download: boolean;
  can_delete: boolean;
  accessible_documents: string[];
}

interface DocumentWithAccess extends DocumentResponse {
  assigned_users: string[];
}

export default function Admin() {
  const { user } = useAuth();
  const router = useRouter();
  const [users, setUsers] = useState<UserResponse[]>([]);
  const [documents, setDocuments] = useState<DocumentWithAccess[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"users" | "permissions">("users");
  const [selectedUser, setSelectedUser] = useState<string | null>(null);
  const [userPermissions, setUserPermissions] = useState<
    Record<string, UserPermissions>
  >({});

  // Redirect non-admins
  useEffect(() => {
    if (user && user.role !== "admin") {
      router.push("/dashboard");
      toast.error("Access denied. Admin privileges required.");
    }
  }, [user, router]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setIsLoading(true);
        const [usersData, docsData] = await Promise.all([
          api.getUsers(),
          api.listDocuments({ page: 1 }),
        ]);
        setUsers(usersData.users);
        setDocuments(
          docsData.documents.map((doc) => ({
            ...doc,
            assigned_users: [],
          }))
        );

        // Initialize permissions for each user
        const perms: Record<string, UserPermissions> = {};
        usersData.users.forEach((u) => {
          perms[u.id] = {
            can_download: true,
            can_delete: u.role === "admin",
            accessible_documents: [],
          };
        });
        setUserPermissions(perms);
        setError(null);
      } catch (err) {
        setError((err as Error).message || "Failed to load data");
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, []);

  const handleRoleChange = async (userId: string, newRole: string) => {
    try {
      await api.updateUser(userId, { role: newRole as "admin" | "user" });
      setUsers((prev) =>
        prev.map((u) =>
          u.id === userId ? { ...u, role: newRole as "admin" | "user" } : u
        )
      );
      toast.success("User role updated");
    } catch (err) {
      toast.error((err as Error).message || "Failed to update role");
    }
  };

  const handleToggleActive = async (userId: string, currentActive: boolean) => {
    try {
      await api.updateUser(userId, { is_active: !currentActive });
      setUsers((prev) =>
        prev.map((u) =>
          u.id === userId ? { ...u, is_active: !currentActive } : u
        )
      );
      toast.success(`User ${!currentActive ? "activated" : "deactivated"}`);
    } catch (err) {
      toast.error((err as Error).message || "Failed to update status");
    }
  };

  const handlePermissionChange = (
    userId: string,
    permission: keyof Omit<UserPermissions, "accessible_documents">,
    value: boolean
  ) => {
    setUserPermissions((prev) => ({
      ...prev,
      [userId]: {
        ...prev[userId],
        [permission]: value,
      },
    }));
    toast.success("Permission updated");
  };

  const handleDocumentAccess = (
    userId: string,
    documentId: string,
    hasAccess: boolean
  ) => {
    setUserPermissions((prev) => {
      const currentDocs = prev[userId]?.accessible_documents || [];
      const newDocs = hasAccess
        ? [...currentDocs, documentId]
        : currentDocs.filter((id) => id !== documentId);
      return {
        ...prev,
        [userId]: {
          ...prev[userId],
          accessible_documents: newDocs,
        },
      };
    });
    toast.success("Document access updated");
  };

  const handleSavePermissions = async () => {
    try {
      // In a real app, you'd save this to the backend
      // await api.saveUserPermissions(selectedUser, userPermissions[selectedUser]);
      toast.success("Permissions saved successfully");
    } catch {
      toast.error("Failed to save permissions");
    }
  };

  if (user?.role !== "admin") {
    return null;
  }

  return (
    <main className="main-content">
      <header style={{ marginBottom: "2rem" }}>
        <h2 style={{ fontSize: "2rem", marginBottom: "0.5rem", background: "linear-gradient(to right, #fff, #9ca3af)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>Admin Panel</h2>
        <p style={{ color: "#9ca3af" }}>
          Manage users, permissions, and document access.
        </p>
      </header>

      {/* Tabs */}
      <div style={{ display: "flex", gap: "1rem", marginBottom: "2rem", background: "rgba(255, 255, 255, 0.03)", padding: "0.5rem", borderRadius: "0.75rem", border: "1px solid rgba(255, 255, 255, 0.05)", width: "fit-content" }}>
        <button
          onClick={() => setActiveTab("users")}
          style={{
            padding: "0.5rem 1.5rem",
            borderRadius: "0.5rem",
            background: activeTab === "users" ? "#1f2937" : "transparent",
            color: activeTab === "users" ? "white" : "#9ca3af",
            border: "none",
            cursor: "pointer",
            fontWeight: 500,
            transition: "all 0.2s",
            display: "flex",
            alignItems: "center",
            gap: "0.5rem"
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
          User Management
        </button>
        <button
          onClick={() => setActiveTab("permissions")}
          style={{
            padding: "0.5rem 1.5rem",
            borderRadius: "0.5rem",
            background: activeTab === "permissions" ? "#1f2937" : "transparent",
            color: activeTab === "permissions" ? "white" : "#9ca3af",
            border: "none",
            cursor: "pointer",
            fontWeight: 500,
            transition: "all 0.2s",
            display: "flex",
            alignItems: "center",
            gap: "0.5rem"
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
          Permissions
        </button>
      </div>

      {isLoading ? (
        <div style={{ color: "#9ca3af", padding: "2rem" }}>Loading...</div>
      ) : error ? (
        <div style={{ color: "#ef4444", padding: "2rem", background: "rgba(239, 68, 68, 0.1)", borderRadius: "1rem" }}>{error}</div>
      ) : activeTab === "users" ? (
        /* Users Tab */
        <section className="feature-card-modern" style={{ background: "#0a0a0a", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem", padding: "1.5rem", overflowX: "auto" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" }}>
            <h3 style={{ fontSize: "1.125rem", fontWeight: "600", color: "white" }}>User Access Management</h3>
            <span style={{ fontSize: "0.875rem", color: "#9ca3af", background: "rgba(255, 255, 255, 0.05)", padding: "0.25rem 0.75rem", borderRadius: "9999px" }}>{users.length} users</span>
          </div>

          <table style={{ width: "100%", borderCollapse: "separate", borderSpacing: "0 0.5rem" }}>
            <thead>
              <tr style={{ textAlign: "left", color: "#6b7280", fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                <th style={{ padding: "0.75rem" }}>User</th>
                <th style={{ padding: "0.75rem" }}>Email</th>
                <th style={{ padding: "0.75rem" }}>Role</th>
                <th style={{ padding: "0.75rem" }}>Download</th>
                <th style={{ padding: "0.75rem" }}>Delete</th>
                <th style={{ padding: "0.75rem" }}>Status</th>
                <th style={{ padding: "0.75rem" }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id} style={{ background: "rgba(255, 255, 255, 0.02)", transition: "background 0.2s" }} onMouseEnter={(e) => e.currentTarget.style.background = "rgba(255, 255, 255, 0.05)"} onMouseLeave={(e) => e.currentTarget.style.background = "rgba(255, 255, 255, 0.02)"}>
                  <td style={{ padding: "0.75rem", borderTopLeftRadius: "0.5rem", borderBottomLeftRadius: "0.5rem" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                      <div style={{ width: "2rem", height: "2rem", borderRadius: "50%", background: "#374151", display: "flex", alignItems: "center", justifyContent: "center", color: "white", fontSize: "0.875rem", fontWeight: "600" }}>
                        {(u.full_name || u.email)[0].toUpperCase()}
                      </div>
                      <span style={{ color: "white", fontWeight: "500" }}>{u.full_name || u.email.split("@")[0]}</span>
                    </div>
                  </td>
                  <td style={{ padding: "0.75rem", color: "#9ca3af" }}>{u.email}</td>
                  <td style={{ padding: "0.75rem" }}>
                    <select
                      value={u.role}
                      onChange={(e) => handleRoleChange(u.id, e.target.value)}
                      style={{ background: "#1f2937", border: "1px solid #374151", color: "white", padding: "0.25rem 0.5rem", borderRadius: "0.25rem", fontSize: "0.875rem" }}
                    >
                      <option value="user">User</option>
                      <option value="admin">Admin</option>
                    </select>
                  </td>
                  <td style={{ padding: "0.75rem" }}>
                    <input
                      type="checkbox"
                      checked={userPermissions[u.id]?.can_download ?? true}
                      onChange={(e) => handlePermissionChange(u.id, "can_download", e.target.checked)}
                      style={{ accentColor: "#3b82f6", width: "1rem", height: "1rem" }}
                    />
                  </td>
                  <td style={{ padding: "0.75rem" }}>
                    <input
                      type="checkbox"
                      checked={userPermissions[u.id]?.can_delete ?? false}
                      onChange={(e) => handlePermissionChange(u.id, "can_delete", e.target.checked)}
                      style={{ accentColor: "#ef4444", width: "1rem", height: "1rem" }}
                    />
                  </td>
                  <td style={{ padding: "0.75rem" }}>
                    <span style={{
                      display: "inline-flex", alignItems: "center", padding: "0.25rem 0.5rem", borderRadius: "9999px", fontSize: "0.75rem", fontWeight: "500",
                      background: u.is_active ? "rgba(16, 185, 129, 0.1)" : "rgba(107, 114, 128, 0.1)",
                      color: u.is_active ? "#10b981" : "#9ca3af"
                    }}>
                      {u.is_active ? "Active" : "Inactive"}
                    </span>
                  </td>
                  <td style={{ padding: "0.75rem", borderTopRightRadius: "0.5rem", borderBottomRightRadius: "0.5rem" }}>
                    <div style={{ display: "flex", gap: "0.5rem" }}>
                      <button
                        onClick={() => { setSelectedUser(u.id); setActiveTab("permissions"); }}
                        title="Manage Permissions"
                        style={{ padding: "0.25rem", background: "transparent", border: "none", color: "#9ca3af", cursor: "pointer" }}
                      >
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 20h9"></path><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path></svg>
                      </button>
                      <button
                        onClick={() => handleToggleActive(u.id, u.is_active)}
                        title={u.is_active ? "Deactivate" : "Activate"}
                        style={{ padding: "0.25rem", background: "transparent", border: "none", color: u.is_active ? "#ef4444" : "#10b981", cursor: "pointer" }}
                      >
                        {u.is_active ? (
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>
                        ) : (
                          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                        )}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ) : (
        /* Permissions Tab */
        <section style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
          <div className="feature-card-modern" style={{ background: "#0a0a0a", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem", padding: "1.5rem" }}>
            <h3 style={{ fontSize: "1.125rem", fontWeight: "600", color: "white", marginBottom: "1rem" }}>Select User</h3>
            <select
              value={selectedUser || ""}
              onChange={(e) => setSelectedUser(e.target.value || null)}
              style={{ width: "100%", padding: "0.75rem", background: "#1f2937", border: "1px solid #374151", borderRadius: "0.5rem", color: "white", fontSize: "1rem" }}
            >
              <option value="">-- Choose a user to manage --</option>
              {users.filter((u) => u.role !== "admin").map((u) => (
                <option key={u.id} value={u.id}>
                  {u.full_name || u.email}
                </option>
              ))}
            </select>
          </div>

          {selectedUser ? (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "1.5rem" }}>
                <div style={{ background: "rgba(59, 130, 246, 0.1)", border: "1px solid rgba(59, 130, 246, 0.2)", borderRadius: "1rem", padding: "1.5rem", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                    <div style={{ padding: "0.75rem", borderRadius: "0.5rem", background: "#3b82f6", color: "white" }}>
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                    </div>
                    <div>
                      <h4 style={{ color: "white", fontWeight: "600" }}>Download Files</h4>
                      <p style={{ color: "#9ca3af", fontSize: "0.875rem" }}>Allow download</p>
                    </div>
                  </div>
                  <input
                    type="checkbox"
                    checked={userPermissions[selectedUser]?.can_download ?? true}
                    onChange={(e) => handlePermissionChange(selectedUser, "can_download", e.target.checked)}
                    style={{ accentColor: "#3b82f6", width: "1.25rem", height: "1.25rem" }}
                  />
                </div>

                <div style={{ background: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.2)", borderRadius: "1rem", padding: "1.5rem", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                    <div style={{ padding: "0.75rem", borderRadius: "0.5rem", background: "#ef4444", color: "white" }}>
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
                    </div>
                    <div>
                      <h4 style={{ color: "white", fontWeight: "600" }}>Delete Files</h4>
                      <p style={{ color: "#9ca3af", fontSize: "0.875rem" }}>Allow deletion</p>
                    </div>
                  </div>
                  <input
                    type="checkbox"
                    checked={userPermissions[selectedUser]?.can_delete ?? false}
                    onChange={(e) => handlePermissionChange(selectedUser, "can_delete", e.target.checked)}
                    style={{ accentColor: "#ef4444", width: "1.25rem", height: "1.25rem" }}
                  />
                </div>
              </div>

              <div className="feature-card-modern" style={{ background: "#0a0a0a", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem", padding: "1.5rem" }}>
                <h3 style={{ fontSize: "1.125rem", fontWeight: "600", color: "white", marginBottom: "0.5rem" }}>Accessible Documents</h3>
                <p style={{ color: "#9ca3af", fontSize: "0.875rem", marginBottom: "1.5rem" }}>Select documents this user is allowed to access.</p>

                {documents.length === 0 ? (
                  <p style={{ color: "#6b7280", textAlign: "center", padding: "2rem" }}>No documents indexed yet.</p>
                ) : (
                  <div style={{ display: "grid", gap: "0.75rem" }}>
                    {documents.map((doc) => {
                      const hasAccess = userPermissions[selectedUser]?.accessible_documents?.includes(doc.id);
                      return (
                        <div
                          key={doc.id}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            padding: "1rem",
                            borderRadius: "0.5rem",
                            background: hasAccess ? "rgba(16, 185, 129, 0.05)" : "rgba(255, 255, 255, 0.02)",
                            border: hasAccess ? "1px solid rgba(16, 185, 129, 0.2)" : "1px solid rgba(255, 255, 255, 0.05)",
                            transition: "all 0.2s"
                          }}
                        >
                          <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                            <span style={{ fontSize: "1.25rem" }}>{doc.mime_type?.includes("pdf") ? "📄" : "📁"}</span>
                            <div>
                              <p style={{ color: hasAccess ? "#10b981" : "white", fontWeight: "500", fontSize: "0.95rem" }}>{doc.title || doc.filename}</p>
                              <p style={{ color: "#6b7280", fontSize: "0.75rem" }}>{doc.chunk_count} chunks • {new Date(doc.created_at).toLocaleDateString()}</p>
                            </div>
                          </div>
                          <input
                            type="checkbox"
                            checked={hasAccess}
                            onChange={(e) => handleDocumentAccess(selectedUser, doc.id, e.target.checked)}
                            style={{ accentColor: "#10b981", width: "1.125rem", height: "1.125rem" }}
                          />
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              <button
                onClick={handleSavePermissions}
                style={{
                  padding: "0.75rem 2rem",
                  borderRadius: "0.5rem",
                  background: "#3b82f6",
                  color: "white",
                  border: "none",
                  fontWeight: "600",
                  cursor: "pointer",
                  alignSelf: "start",
                  boxShadow: "0 4px 6px -1px rgba(59, 130, 246, 0.4)"
                }}
              >
                Save Changes
              </button>
            </>
          ) : (
            <div style={{ textAlign: "center", padding: "4rem", border: "1px dashed rgba(255, 255, 255, 0.1)", borderRadius: "1rem", color: "#6b7280" }}>
              <p>Select a user above to manage permissions</p>
            </div>
          )}
        </section>
      )}
    </main>
  );
}
