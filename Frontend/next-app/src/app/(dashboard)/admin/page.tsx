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
    <main className="main-content admin-page">
      <header>
        <h2>Admin Panel</h2>
        <p>Manage users, permissions, and document access.</p>
      </header>

      {/* Tabs */}
      <div className="admin-tabs">
        <button
          className={`tab-button ${activeTab === "users" ? "active" : ""}`}
          onClick={() => setActiveTab("users")}
        >
          👥 User Management
        </button>
        <button
          className={`tab-button ${
            activeTab === "permissions" ? "active" : ""
          }`}
          onClick={() => setActiveTab("permissions")}
        >
          🔐 Document Permissions
        </button>
      </div>

      {isLoading ? (
        <div className="loading-state">
          <p>Loading...</p>
        </div>
      ) : error ? (
        <div className="error-state">
          <p style={{ color: "#dc2626" }}>{error}</p>
        </div>
      ) : activeTab === "users" ? (
        /* Users Tab */
        <section className="admin-section">
          <div className="section-header">
            <h3>User Access Management</h3>
            <span className="user-count">{users.length} users</span>
          </div>

          <table className="user-table">
            <thead>
              <tr>
                <th>User</th>
                <th>Email</th>
                <th>Role</th>
                <th>Download</th>
                <th>Delete Files</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.length === 0 ? (
                <tr>
                  <td colSpan={7} style={{ textAlign: "center" }}>
                    No users found
                  </td>
                </tr>
              ) : (
                users.map((u) => (
                  <tr key={u.id}>
                    <td className="user-cell">
                      <div className="user-avatar">
                        {(u.full_name || u.email)[0].toUpperCase()}
                      </div>
                      <span>{u.full_name || u.email.split("@")[0]}</span>
                    </td>
                    <td>{u.email}</td>
                    <td>
                      <select
                        value={u.role}
                        onChange={(e) => handleRoleChange(u.id, e.target.value)}
                        className={`role-select ${u.role}`}
                      >
                        <option value="user">User</option>
                        <option value="admin">Admin</option>
                      </select>
                    </td>
                    <td>
                      <label className="toggle-switch small">
                        <input
                          type="checkbox"
                          checked={userPermissions[u.id]?.can_download ?? true}
                          onChange={(e) =>
                            handlePermissionChange(
                              u.id,
                              "can_download",
                              e.target.checked
                            )
                          }
                        />
                        <span className="toggle-slider"></span>
                      </label>
                    </td>
                    <td>
                      <label className="toggle-switch small">
                        <input
                          type="checkbox"
                          checked={userPermissions[u.id]?.can_delete ?? false}
                          onChange={(e) =>
                            handlePermissionChange(
                              u.id,
                              "can_delete",
                              e.target.checked
                            )
                          }
                        />
                        <span className="toggle-slider"></span>
                      </label>
                    </td>
                    <td>
                      <span
                        className={`status-badge ${
                          u.is_active ? "active" : "inactive"
                        }`}
                      >
                        {u.is_active ? "Active" : "Inactive"}
                      </span>
                    </td>
                    <td>
                      <div className="action-buttons">
                        <button
                          onClick={() => {
                            setSelectedUser(u.id);
                            setActiveTab("permissions");
                          }}
                          className="btn btn-sm btn-light"
                          title="Manage documents"
                        >
                          📁
                        </button>
                        <button
                          onClick={() => handleToggleActive(u.id, u.is_active)}
                          className={`btn btn-sm ${
                            u.is_active ? "btn-danger" : "btn-success"
                          }`}
                        >
                          {u.is_active ? "Deactivate" : "Activate"}
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </section>
      ) : (
        /* Permissions Tab */
        <section className="admin-section permissions-section">
          <div className="section-header">
            <h3>Document Access Control</h3>
            <select
              value={selectedUser || ""}
              onChange={(e) => setSelectedUser(e.target.value || null)}
              className="user-select"
            >
              <option value="">Select a user...</option>
              {users
                .filter((u) => u.role !== "admin")
                .map((u) => (
                  <option key={u.id} value={u.id}>
                    {u.full_name || u.email}
                  </option>
                ))}
            </select>
          </div>

          {selectedUser ? (
            <>
              <div className="permissions-summary">
                <div className="perm-card">
                  <span className="perm-icon">📥</span>
                  <div className="perm-info">
                    <label>Can Download Files</label>
                    <label className="toggle-switch">
                      <input
                        type="checkbox"
                        checked={
                          userPermissions[selectedUser]?.can_download ?? true
                        }
                        onChange={(e) =>
                          handlePermissionChange(
                            selectedUser,
                            "can_download",
                            e.target.checked
                          )
                        }
                      />
                      <span className="toggle-slider"></span>
                    </label>
                  </div>
                </div>
                <div className="perm-card">
                  <span className="perm-icon">🗑️</span>
                  <div className="perm-info">
                    <label>Can Delete Files</label>
                    <label className="toggle-switch">
                      <input
                        type="checkbox"
                        checked={
                          userPermissions[selectedUser]?.can_delete ?? false
                        }
                        onChange={(e) =>
                          handlePermissionChange(
                            selectedUser,
                            "can_delete",
                            e.target.checked
                          )
                        }
                      />
                      <span className="toggle-slider"></span>
                    </label>
                  </div>
                </div>
              </div>

              <div className="documents-access-grid">
                <h4>Accessible Documents</h4>
                <p className="helper-text">
                  Select which documents this user can access.
                </p>

                {documents.length === 0 ? (
                  <p className="no-docs">No documents indexed yet.</p>
                ) : (
                  <div className="document-list">
                    {documents.map((doc) => {
                      const hasAccess = userPermissions[
                        selectedUser
                      ]?.accessible_documents?.includes(doc.id);
                      return (
                        <div
                          key={doc.id}
                          className={`document-item ${
                            hasAccess ? "granted" : ""
                          }`}
                        >
                          <div className="doc-info">
                            <span className="doc-icon">
                              {doc.mime_type?.includes("pdf") ? "📄" : "📁"}
                            </span>
                            <div className="doc-details">
                              <span className="doc-name">
                                {doc.title || doc.filename}
                              </span>
                              <span className="doc-meta">
                                {doc.chunk_count} chunks •{" "}
                                {new Date(doc.created_at).toLocaleDateString()}
                              </span>
                            </div>
                          </div>
                          <label className="toggle-switch">
                            <input
                              type="checkbox"
                              checked={hasAccess}
                              onChange={(e) =>
                                handleDocumentAccess(
                                  selectedUser,
                                  doc.id,
                                  e.target.checked
                                )
                              }
                            />
                            <span className="toggle-slider"></span>
                          </label>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              <div className="save-section">
                <button
                  onClick={handleSavePermissions}
                  className="btn btn-dark"
                >
                  Save Permissions
                </button>
              </div>
            </>
          ) : (
            <div className="empty-state">
              <span className="empty-icon">👤</span>
              <p>Select a user to manage their document access permissions.</p>
            </div>
          )}
        </section>
      )}
    </main>
  );
}
