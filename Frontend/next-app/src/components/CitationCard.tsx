"use client";

import { useState } from "react";

export interface CitationSource {
  id: string;
  name: string;
  page?: number;
  section?: string;
  text?: string;
  relevance: "High" | "Medium" | "Low";
  document_id?: string;
}

interface CitationCardProps {
  source: CitationSource;
  onNavigate?: (documentId: string) => void;
  onPreview?: (chunkId: string) => void;
  compact?: boolean;
}

export function CitationCard({
  source,
  onNavigate,
  onPreview,
  compact = false,
}: CitationCardProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    const citation = `${source.name}${
      source.page ? `, p. ${source.page}` : ""
    }${source.section ? ` (${source.section})` : ""}`;
    await navigator.clipboard.writeText(citation);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getRelevanceColor = () => {
    switch (source.relevance) {
      case "High":
        return { bg: "#065f46", text: "#10b981" };
      case "Medium":
        return { bg: "#92400e", text: "#f59e0b" };
      case "Low":
        return { bg: "#374151", text: "#9ca3af" };
    }
  };

  const colors = getRelevanceColor();

  if (compact) {
    return (
      <span
        className="citation-badge"
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "0.25rem",
          padding: "0.125rem 0.5rem",
          borderRadius: "9999px",
          backgroundColor: colors.bg,
          color: colors.text,
          fontSize: "0.75rem",
          cursor: "pointer",
        }}
        onClick={() => onPreview?.(source.id)}
        title={source.text?.substring(0, 100) || source.name}
      >
        📄 {source.name}
        {source.page && ` p.${source.page}`}
      </span>
    );
  }

  return (
    <div
      className="citation-card"
      style={{
        backgroundColor: "#1f2937",
        border: "1px solid #374151",
        borderRadius: "0.5rem",
        padding: "1rem",
        marginBottom: "0.5rem",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          marginBottom: "0.5rem",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span style={{ fontSize: "1.25rem" }}>📄</span>
          <div>
            <h4
              style={{
                margin: 0,
                color: "#f9fafb",
                fontSize: "0.875rem",
                fontWeight: 600,
              }}
            >
              {source.name}
            </h4>
            <div style={{ fontSize: "0.75rem", color: "#9ca3af" }}>
              {source.page && <span>Page {source.page}</span>}
              {source.section && <span> • {source.section}</span>}
            </div>
          </div>
        </div>

        {/* Relevance Badge */}
        <span
          style={{
            backgroundColor: colors.bg,
            color: colors.text,
            padding: "0.125rem 0.5rem",
            borderRadius: "9999px",
            fontSize: "0.625rem",
            fontWeight: 500,
            textTransform: "uppercase",
          }}
        >
          {source.relevance}
        </span>
      </div>

      {/* Text Preview */}
      {source.text && (
        <p
          style={{
            margin: "0.5rem 0",
            color: "#d1d5db",
            fontSize: "0.8125rem",
            lineHeight: 1.5,
            overflow: "hidden",
            display: "-webkit-box",
            WebkitLineClamp: 3,
            WebkitBoxOrient: "vertical",
          }}
        >
          {source.text}
        </p>
      )}

      {/* Actions */}
      <div
        style={{
          display: "flex",
          gap: "0.5rem",
          marginTop: "0.75rem",
        }}
      >
        <button
          onClick={handleCopy}
          style={{
            padding: "0.375rem 0.75rem",
            borderRadius: "0.25rem",
            backgroundColor: "#374151",
            color: "#f9fafb",
            border: "none",
            cursor: "pointer",
            fontSize: "0.75rem",
            display: "flex",
            alignItems: "center",
            gap: "0.25rem",
          }}
        >
          {copied ? "✓ Copied" : "📋 Copy Citation"}
        </button>

        {source.document_id && onNavigate && (
          <button
            onClick={() => onNavigate(source.document_id!)}
            style={{
              padding: "0.375rem 0.75rem",
              borderRadius: "0.25rem",
              backgroundColor: "#3b82f6",
              color: "#ffffff",
              border: "none",
              cursor: "pointer",
              fontSize: "0.75rem",
            }}
          >
            View Document
          </button>
        )}

        {onPreview && (
          <button
            onClick={() => onPreview(source.id)}
            style={{
              padding: "0.375rem 0.75rem",
              borderRadius: "0.25rem",
              backgroundColor: "transparent",
              color: "#60a5fa",
              border: "1px solid #3b82f6",
              cursor: "pointer",
              fontSize: "0.75rem",
            }}
          >
            Preview
          </button>
        )}
      </div>
    </div>
  );
}

export default CitationCard;
