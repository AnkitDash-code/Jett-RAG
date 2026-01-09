"use client";

import { useState, useEffect, useCallback } from "react";
import { api } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { useDropzone } from "react-dropzone";
import type { DocumentResponse } from "@/types/api";

export default function DocumentsPage() {
  const { user } = useAuth();
  const router = useRouter();
  const [documents, setDocuments] = useState<DocumentResponse[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [filter, setFilter] = useState<
    "all" | "indexed" | "processing" | "error" | "indexed_light"
  >("all");

  // Redirect non-admins
  useEffect(() => {
    if (user && user.role !== "admin") {
      router.push("/dashboard");
      toast.error("Access denied. Admin privileges required.");
    }
  }, [user, router]);

  const fetchDocuments = useCallback(async () => {
    try {
      setIsLoading(true);
      const data = await api.listDocuments({ page: 1 });
      setDocuments(data.documents);
    } catch (err) {
      toast.error((err as Error).message || "Failed to load documents");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      if (acceptedFiles.length === 0) return;

      setIsUploading(true);
      try {
        for (const file of acceptedFiles) {
          await api.uploadDocument(file);
          toast.success(`Uploaded: ${file.name}`);
        }
        // Refresh the list
        await fetchDocuments();
      } catch (err) {
        toast.error((err as Error).message || "Upload failed");
      } finally {
        setIsUploading(false);
      }
    },
    [fetchDocuments]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "text/plain": [".txt"],
      "text/markdown": [".md"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        [".docx"],
    },
  });

  const handleDelete = async (docId: string, filename: string) => {
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) return;

    try {
      await api.deleteDocument(docId);
      setDocuments((prev) => prev.filter((d) => d.id !== docId));
      toast.success("Document deleted");
    } catch (err) {
      toast.error((err as Error).message || "Failed to delete");
    }
  };

  const filteredDocuments = documents.filter((doc) => {
    if (filter === "all") return true;
    return doc.status === filter;
  });

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "indexed":
        return "✅";
      case "processing":
        return "⏳";
      case "error":
        return "❌";
      default:
        return "📄";
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  if (user?.role !== "admin") {
    return null;
  }

  return (
    <main className="main-content">
      <header style={{ marginBottom: "2rem" }}>
        <h2 style={{ fontSize: "2rem", marginBottom: "0.5rem", background: "linear-gradient(to right, #fff, #9ca3af)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>Documents</h2>
        <p style={{ color: "#9ca3af" }}>
          Upload and manage your knowledge base sources.
        </p>
      </header>

      {/* Stats Bar */}
      <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem", marginBottom: "2rem" }}>
        <div style={{ background: "rgba(255, 255, 255, 0.06)", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem", padding: "1.25rem" }}>
          <p style={{ color: "#9ca3af", fontSize: "0.875rem", marginBottom: "0.5rem" }}>Total Documents</p>
          <p style={{ color: "white", fontSize: "1.5rem", fontWeight: "600" }}>{documents.length}</p>
        </div>
        <div style={{ background: "rgba(16, 185, 129, 0.2)", border: "1px solid rgba(16, 185, 129, 0.2)", borderRadius: "1rem", padding: "1.25rem" }}>
          <p style={{ color: "#34d399", fontSize: "0.875rem", marginBottom: "0.5rem" }}>Indexed</p>
          <p style={{ color: "white", fontSize: "1.5rem", fontWeight: "600" }}>{documents.filter((d) => d.status === "indexed").length}</p>
        </div>
        <div style={{ background: "rgba(245, 158, 11, 0.2)", border: "1px solid rgba(245, 158, 11, 0.2)", borderRadius: "1rem", padding: "1.25rem" }}>
          <p style={{ color: "#fbbf24", fontSize: "0.875rem", marginBottom: "0.5rem" }}>Processing</p>
          <p style={{ color: "white", fontSize: "1.5rem", fontWeight: "600" }}>{documents.filter((d) => d.status === "processing").length}</p>
        </div>
        <div style={{ background: "rgba(59, 130, 246, 0.2)", border: "1px solid rgba(59, 130, 246, 0.2)", borderRadius: "1rem", padding: "1.25rem" }}>
          <p style={{ color: "#60a5fa", fontSize: "0.875rem", marginBottom: "0.5rem" }}>Total Chunks</p>
          <p style={{ color: "white", fontSize: "1.5rem", fontWeight: "600" }}>{documents.reduce((acc, d) => acc + d.chunk_count, 0)}</p>
        </div>
      </section>

      {/* Upload Section */}
      <section style={{ marginBottom: "2rem" }}>
        <div
          {...getRootProps()}
          style={{
            border: isDragActive ? "2px dashed #3b82f6" : "2px dashed rgba(255, 255, 255, 0.1)",
            borderRadius: "1rem",
            padding: "3rem",
            textAlign: "center",
            background: isDragActive ? "rgba(59, 130, 246, 0.05)" : "rgba(0, 0, 0, 0.2)",
            cursor: "pointer",
            transition: "all 0.2s ease"
          }}
        >
          <input {...getInputProps()} />
          <div style={{ marginBottom: "1rem", display: "flex", justifyContent: "center" }}>
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke={isDragActive ? "#3b82f6" : "#6b7280"} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>
          <p style={{ color: "white", fontWeight: "500", marginBottom: "0.5rem" }}>
            {isUploading ? "Uploading..." : isDragActive ? "Drop files here..." : "Click or drag files to upload"}
          </p>
          <p style={{ color: "#6b7280", fontSize: "0.875rem" }}>
            Supports PDF, TXT, MD, DOCX (Max 10MB)
          </p>
        </div>
      </section>

      {/* Filter & Document List */}
      <section>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" }}>
          <h3 style={{ fontSize: "1.25rem", fontWeight: "600", color: "white" }}>All Documents</h3>
          <div style={{ display: "flex", gap: "0.5rem", background: "#1f2937", padding: "0.25rem", borderRadius: "0.5rem" }}>
            {(["all", "indexed", "indexed_light", "processing", "error"] as const).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                style={{
                  padding: "0.375rem 0.75rem",
                  borderRadius: "0.375rem",
                  fontSize: "0.875rem",
                  fontWeight: "500",
                  textTransform: "capitalize",
                  background: filter === f ? "#374151" : "transparent",
                  color: filter === f ? "white" : "#9ca3af",
                  border: "none",
                  cursor: "pointer",
                  transition: "all 0.2s"
                }}
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        {isLoading ? (
          <div style={{ padding: "3rem", textAlign: "center", color: "#6b7280" }}>Loading documents...</div>
        ) : filteredDocuments.length === 0 ? (
          <div style={{ padding: "4rem", textAlign: "center", border: "1px dashed rgba(255, 255, 255, 0.1)", borderRadius: "1rem" }}>
            <p style={{ color: "#9ca3af" }}>No documents found matching your filter.</p>
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: "1.5rem" }}>
            {filteredDocuments.map((doc) => (
              <div
                key={doc.id}
                style={{
                  background: "#0a0a0a",
                  border: "1px solid rgba(255, 255, 255, 0.1)",
                  borderRadius: "1rem",
                  padding: "1.5rem",
                  display: "flex",
                  flexDirection: "column",
                  transition: "transform 0.2s",
                  position: "relative",
                  overflow: "hidden"
                }}
                className="feature-card-modern"
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: "1rem" }}>
                  <div style={{
                    width: "40px",
                    height: "40px",
                    borderRadius: "8px",
                    background: doc.status === 'indexed' ? "rgba(16, 185, 129, 0.1)" : doc.status === 'error' ? "rgba(239, 68, 68, 0.1)" : "rgba(245, 158, 11, 0.1)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: doc.status === 'indexed' ? "#10b981" : doc.status === 'error' ? "#ef4444" : "#f59e0b"
                  }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                      <polyline points="14 2 14 8 20 8" />
                    </svg>
                  </div>
                  <span style={{
                    fontSize: "0.75rem",
                    padding: "0.25rem 0.5rem",
                    borderRadius: "9999px",
                    background: doc.status === 'indexed' ? "rgba(16, 185, 129, 0.1)" : doc.status === 'error' ? "rgba(239, 68, 68, 0.1)" : "rgba(245, 158, 11, 0.1)",
                    color: doc.status === 'indexed' ? "#10b981" : doc.status === 'error' ? "#ef4444" : "#f59e0b"
                  }}>
                    {doc.status}
                  </span>
                </div>

                <h4 style={{ color: "white", fontSize: "1rem", fontWeight: "600", marginBottom: "0.25rem", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={doc.title || doc.filename}>
                  {doc.title || doc.filename}
                </h4>
                <p style={{ color: "#6b7280", fontSize: "0.875rem", marginBottom: "1rem", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {doc.original_filename}
                </p>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", marginBottom: "1rem" }}>
                  <div>
                    <p style={{ color: "#6b7280", fontSize: "0.75rem" }}>Size</p>
                    <p style={{ color: "#d1d5db", fontSize: "0.875rem" }}>{formatFileSize(doc.file_size)}</p>
                  </div>
                  <div>
                    <p style={{ color: "#6b7280", fontSize: "0.75rem" }}>Chunks</p>
                    <p style={{ color: "#d1d5db", fontSize: "0.875rem" }}>{doc.chunk_count}</p>
                  </div>
                  <div>
                    <p style={{ color: "#6b7280", fontSize: "0.75rem" }}>Date</p>
                    <p style={{ color: "#d1d5db", fontSize: "0.875rem" }}>{new Date(doc.created_at).toLocaleDateString()}</p>
                  </div>
                </div>

                <div style={{ marginTop: "auto", borderTop: "1px solid rgba(255, 255, 255, 0.05)", paddingTop: "1rem", display: "flex", justifyContent: "flex-end" }}>
                  <button
                    onClick={() => handleDelete(doc.id, doc.filename)}
                    style={{
                      background: "transparent",
                      border: "none",
                      color: "#ef4444",
                      fontSize: "0.875rem",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "0.25rem",
                      padding: "0.25rem 0.5rem",
                      borderRadius: "0.25rem",
                      transition: "background 0.2s"
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = "rgba(239, 68, 68, 0.1)"}
                    onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="3 6 5 6 21 6" />
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                    </svg>
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
