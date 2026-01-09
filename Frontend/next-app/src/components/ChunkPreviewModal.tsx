"use client";

import { useState, useEffect } from "react";
import { api } from "@/lib/api";

export interface ChunkPreview {
  id: string;
  content: string;
  document_id: string;
  document_name: string;
  chunk_index: number;
  total_chunks: number;
  page?: number;
  section?: string;
  word_count: number;
  entities?: Array<{
    name: string;
    type: string;
  }>;
  surrounding_context?: {
    prev_chunk?: string;
    next_chunk?: string;
  };
}

interface ChunkPreviewModalProps {
  chunkId: string | null;
  onClose: () => void;
  onNavigate?: (direction: "prev" | "next", currentChunkId: string) => void;
}

export function ChunkPreviewModal({
  chunkId,
  onClose,
  onNavigate,
}: ChunkPreviewModalProps) {
  const [chunk, setChunk] = useState<ChunkPreview | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showContext, setShowContext] = useState(false);

  useEffect(() => {
    if (!chunkId) {
      setChunk(null);
      return;
    }

    const fetchChunk = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await fetch(
          `${
            process.env.NEXT_PUBLIC_API_URL || "http://localhost:8081/v1"
          }/documents/chunks/${chunkId}/preview?include_context=true`,
          {
            headers: {
              Authorization: `Bearer ${api.getAccessToken()}`,
            },
          }
        );

        if (!response.ok) {
          throw new Error("Failed to fetch chunk preview");
        }

        const data = await response.json();
        setChunk(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    };

    fetchChunk();
  }, [chunkId]);

  if (!chunkId) return null;

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0, 0, 0, 0.6)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
        padding: "1rem",
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: "#1f2937",
          borderRadius: "0.75rem",
          width: "100%",
          maxWidth: "800px",
          maxHeight: "90vh",
          display: "flex",
          flexDirection: "column",
          border: "1px solid #374151",
          boxShadow: "0 25px 50px -12px rgba(0, 0, 0, 0.5)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          style={{
            padding: "1rem 1.5rem",
            borderBottom: "1px solid #374151",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div>
            <h3 style={{ margin: 0, color: "#f9fafb", fontSize: "1.125rem" }}>
              📄 {chunk?.document_name || "Loading..."}
            </h3>
            {chunk && (
              <div
                style={{
                  fontSize: "0.875rem",
                  color: "#9ca3af",
                  marginTop: "0.25rem",
                }}
              >
                Chunk {chunk.chunk_index + 1} of {chunk.total_chunks}
                {chunk.page && ` • Page ${chunk.page}`}
                {chunk.section && ` • ${chunk.section}`}
                <span style={{ marginLeft: "0.5rem" }}>
                  ({chunk.word_count} words)
                </span>
              </div>
            )}
          </div>
          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "none",
              color: "#9ca3af",
              cursor: "pointer",
              fontSize: "1.5rem",
              padding: "0.25rem",
              lineHeight: 1,
            }}
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div
          style={{
            flex: 1,
            overflow: "auto",
            padding: "1.5rem",
          }}
        >
          {loading && (
            <div
              style={{ textAlign: "center", padding: "2rem", color: "#9ca3af" }}
            >
              <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>⏳</div>
              Loading chunk preview...
            </div>
          )}

          {error && (
            <div
              style={{
                textAlign: "center",
                padding: "2rem",
                color: "#ef4444",
                backgroundColor: "rgba(239, 68, 68, 0.1)",
                borderRadius: "0.5rem",
              }}
            >
              <div style={{ fontSize: "2rem", marginBottom: "0.5rem" }}>❌</div>
              {error}
            </div>
          )}

          {chunk && !loading && !error && (
            <>
              {/* Previous Context */}
              {showContext && chunk.surrounding_context?.prev_chunk && (
                <div
                  style={{
                    marginBottom: "1rem",
                    padding: "1rem",
                    backgroundColor: "#111827",
                    borderRadius: "0.5rem",
                    border: "1px dashed #374151",
                    opacity: 0.7,
                  }}
                >
                  <div
                    style={{
                      fontSize: "0.75rem",
                      color: "#6b7280",
                      marginBottom: "0.5rem",
                    }}
                  >
                    ↑ Previous Chunk
                  </div>
                  <p
                    style={{
                      color: "#9ca3af",
                      fontSize: "0.875rem",
                      margin: 0,
                    }}
                  >
                    {chunk.surrounding_context.prev_chunk}
                  </p>
                </div>
              )}

              {/* Main Content */}
              <div
                style={{
                  backgroundColor: "#111827",
                  padding: "1.25rem",
                  borderRadius: "0.5rem",
                  border: "1px solid #374151",
                }}
              >
                <p
                  style={{
                    color: "#e5e7eb",
                    fontSize: "0.9375rem",
                    lineHeight: 1.7,
                    margin: 0,
                    whiteSpace: "pre-wrap",
                  }}
                >
                  {chunk.content}
                </p>
              </div>

              {/* Next Context */}
              {showContext && chunk.surrounding_context?.next_chunk && (
                <div
                  style={{
                    marginTop: "1rem",
                    padding: "1rem",
                    backgroundColor: "#111827",
                    borderRadius: "0.5rem",
                    border: "1px dashed #374151",
                    opacity: 0.7,
                  }}
                >
                  <div
                    style={{
                      fontSize: "0.75rem",
                      color: "#6b7280",
                      marginBottom: "0.5rem",
                    }}
                  >
                    ↓ Next Chunk
                  </div>
                  <p
                    style={{
                      color: "#9ca3af",
                      fontSize: "0.875rem",
                      margin: 0,
                    }}
                  >
                    {chunk.surrounding_context.next_chunk}
                  </p>
                </div>
              )}

              {/* Entities */}
              {chunk.entities && chunk.entities.length > 0 && (
                <div style={{ marginTop: "1rem" }}>
                  <h4
                    style={{
                      color: "#9ca3af",
                      fontSize: "0.875rem",
                      marginBottom: "0.5rem",
                    }}
                  >
                    Entities Mentioned:
                  </h4>
                  <div
                    style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}
                  >
                    {chunk.entities.map((entity, idx) => (
                      <span
                        key={idx}
                        style={{
                          padding: "0.25rem 0.75rem",
                          borderRadius: "9999px",
                          backgroundColor: getEntityColor(entity.type),
                          color: "#fff",
                          fontSize: "0.75rem",
                        }}
                      >
                        {entity.name}
                        <span style={{ opacity: 0.7, marginLeft: "0.25rem" }}>
                          ({entity.type})
                        </span>
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div
          style={{
            padding: "1rem 1.5rem",
            borderTop: "1px solid #374151",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <button
              onClick={() => setShowContext(!showContext)}
              style={{
                padding: "0.5rem 1rem",
                borderRadius: "0.375rem",
                backgroundColor: showContext ? "#3b82f6" : "#374151",
                color: "#fff",
                border: "none",
                cursor: "pointer",
                fontSize: "0.875rem",
              }}
            >
              {showContext ? "Hide Context" : "Show Context"}
            </button>
          </div>

          <div style={{ display: "flex", gap: "0.5rem" }}>
            {onNavigate && chunk && chunk.chunk_index > 0 && (
              <button
                onClick={() => onNavigate("prev", chunkId)}
                style={{
                  padding: "0.5rem 1rem",
                  borderRadius: "0.375rem",
                  backgroundColor: "#374151",
                  color: "#fff",
                  border: "none",
                  cursor: "pointer",
                  fontSize: "0.875rem",
                }}
              >
                ← Previous
              </button>
            )}
            {onNavigate &&
              chunk &&
              chunk.chunk_index < chunk.total_chunks - 1 && (
                <button
                  onClick={() => onNavigate("next", chunkId)}
                  style={{
                    padding: "0.5rem 1rem",
                    borderRadius: "0.375rem",
                    backgroundColor: "#374151",
                    color: "#fff",
                    border: "none",
                    cursor: "pointer",
                    fontSize: "0.875rem",
                  }}
                >
                  Next →
                </button>
              )}
            <button
              onClick={onClose}
              style={{
                padding: "0.5rem 1rem",
                borderRadius: "0.375rem",
                backgroundColor: "#ef4444",
                color: "#fff",
                border: "none",
                cursor: "pointer",
                fontSize: "0.875rem",
              }}
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function getEntityColor(type: string): string {
  const colors: Record<string, string> = {
    PERSON: "#3b82f6",
    ORGANIZATION: "#8b5cf6",
    LOCATION: "#10b981",
    DATE: "#f59e0b",
    EVENT: "#ef4444",
    PRODUCT: "#ec4899",
    TECHNOLOGY: "#06b6d4",
    CONCEPT: "#6366f1",
  };
  return colors[type.toUpperCase()] || "#6b7280";
}

export default ChunkPreviewModal;
