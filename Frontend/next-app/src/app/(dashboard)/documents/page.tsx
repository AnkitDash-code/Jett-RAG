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
    "all" | "indexed" | "processing" | "error"
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
    <main className="main-content documents-page">
      <header>
        <h2>Document Management</h2>
        <p>Upload, manage, and monitor document indexing.</p>
      </header>

      {/* Upload Section */}
      <section className="upload-section">
        <div
          {...getRootProps()}
          className={`upload-dropzone ${isDragActive ? "active" : ""} ${
            isUploading ? "uploading" : ""
          }`}
        >
          <input {...getInputProps()} />
          {isUploading ? (
            <>
              <span className="upload-icon">⏳</span>
              <p>Uploading...</p>
            </>
          ) : isDragActive ? (
            <>
              <span className="upload-icon">📥</span>
              <p>Drop files here...</p>
            </>
          ) : (
            <>
              <span className="upload-icon">📁</span>
              <p>Drag & drop files here, or click to select</p>
              <small>Supports PDF, TXT, MD, DOCX</small>
            </>
          )}
        </div>
      </section>

      {/* Stats Bar */}
      <section className="stats-bar">
        <div className="stat-item">
          <span className="stat-value">{documents.length}</span>
          <span className="stat-label">Total Documents</span>
        </div>
        <div className="stat-item">
          <span className="stat-value">
            {documents.filter((d) => d.status === "indexed").length}
          </span>
          <span className="stat-label">Indexed</span>
        </div>
        <div className="stat-item">
          <span className="stat-value">
            {documents.filter((d) => d.status === "processing").length}
          </span>
          <span className="stat-label">Processing</span>
        </div>
        <div className="stat-item">
          <span className="stat-value">
            {documents.reduce((acc, d) => acc + d.chunk_count, 0)}
          </span>
          <span className="stat-label">Total Chunks</span>
        </div>
      </section>

      {/* Filter & Document List */}
      <section className="documents-section">
        <div className="section-header">
          <h3>All Documents</h3>
          <div className="filter-buttons">
            {(["all", "indexed", "processing", "error"] as const).map((f) => (
              <button
                key={f}
                className={`filter-btn ${filter === f ? "active" : ""}`}
                onClick={() => setFilter(f)}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {isLoading ? (
          <div className="loading-state">
            <p>Loading documents...</p>
          </div>
        ) : filteredDocuments.length === 0 ? (
          <div className="empty-state">
            <span className="empty-icon">📭</span>
            <p>No documents found</p>
          </div>
        ) : (
          <div className="documents-grid">
            {filteredDocuments.map((doc) => (
              <div key={doc.id} className={`document-card ${doc.status}`}>
                <div className="doc-header">
                  <span className="doc-status-icon">
                    {getStatusIcon(doc.status)}
                  </span>
                  <span className={`doc-status-badge ${doc.status}`}>
                    {doc.status}
                  </span>
                </div>
                <div className="doc-body">
                  <h4 className="doc-title">{doc.title || doc.filename}</h4>
                  <p className="doc-filename">
                    {doc.original_filename || doc.filename}
                  </p>
                  <div className="doc-meta-grid">
                    <div className="meta-item">
                      <span className="meta-label">Size</span>
                      <span className="meta-value">
                        {formatFileSize(doc.file_size)}
                      </span>
                    </div>
                    <div className="meta-item">
                      <span className="meta-label">Chunks</span>
                      <span className="meta-value">{doc.chunk_count}</span>
                    </div>
                    <div className="meta-item">
                      <span className="meta-label">Pages</span>
                      <span className="meta-value">
                        {doc.page_count || "—"}
                      </span>
                    </div>
                    <div className="meta-item">
                      <span className="meta-label">Access</span>
                      <span className="meta-value">{doc.access_level}</span>
                    </div>
                  </div>
                  <div className="doc-tags">
                    {doc.tags.map((tag) => (
                      <span key={tag} className="tag">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="doc-footer">
                  <span className="doc-date">
                    {new Date(doc.created_at).toLocaleDateString()}
                  </span>
                  <div className="doc-actions">
                    <button
                      onClick={() => handleDelete(doc.id, doc.filename)}
                      className="btn btn-sm btn-danger"
                      title="Delete document"
                    >
                      🗑️ Delete
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}
