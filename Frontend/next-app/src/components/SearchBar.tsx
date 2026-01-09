"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { api, API_BASE_URL } from "@/lib/api";

interface SearchSuggestion {
  entities: Array<{ name: string; type: string }>;
  documents: Array<{ id: string; title: string }>;
  recent_queries: string[];
  popular_topics: string[];
}

interface SearchBarProps {
  onSearch: (query: string) => void;
  onSelectDocument?: (docId: string) => void;
  placeholder?: string;
  autoFocus?: boolean;
}

export function SearchBar({
  onSearch,
  onSelectDocument,
  placeholder = "Search...",
  autoFocus = false,
}: SearchBarProps) {
  const [query, setQuery] = useState("");
  const [suggestions, setSuggestions] = useState<SearchSuggestion | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Flatten suggestions for keyboard navigation
  const flatSuggestions = suggestions
    ? [
        ...suggestions.entities.map((e) => ({
          type: "entity",
          label: e.name,
          meta: e.type,
        })),
        ...suggestions.documents.map((d) => ({
          type: "document",
          label: d.title,
          id: d.id,
        })),
        ...suggestions.recent_queries.map((q) => ({
          type: "recent",
          label: q,
        })),
        ...suggestions.popular_topics.map((t) => ({ type: "topic", label: t })),
      ]
    : [];

  // Fetch suggestions
  const fetchSuggestions = useCallback(async (q: string) => {
    if (q.length < 1) {
      setSuggestions(null);
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(
        `${API_BASE_URL}/search/suggestions?q=${encodeURIComponent(q)}`,
        {
          headers: {
            Authorization: `Bearer ${api.getAccessToken()}`,
          },
        }
      );

      if (response.ok) {
        const data = await response.json();
        setSuggestions(data);
      }
    } catch (err) {
      console.error("Failed to fetch suggestions:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Debounce input
  useEffect(() => {
    const timer = setTimeout(() => {
      fetchSuggestions(query);
    }, 200);

    return () => clearTimeout(timer);
  }, [query, fetchSuggestions]);

  // Click outside to close
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node) &&
        inputRef.current &&
        !inputRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Handle keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!isOpen || flatSuggestions.length === 0) {
      if (e.key === "Enter") {
        onSearch(query);
      }
      return;
    }

    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setSelectedIndex((prev) =>
          prev < flatSuggestions.length - 1 ? prev + 1 : prev
        );
        break;
      case "ArrowUp":
        e.preventDefault();
        setSelectedIndex((prev) => (prev > 0 ? prev - 1 : -1));
        break;
      case "Enter":
        e.preventDefault();
        if (selectedIndex >= 0 && selectedIndex < flatSuggestions.length) {
          handleSelect(flatSuggestions[selectedIndex]);
        } else {
          onSearch(query);
        }
        setIsOpen(false);
        break;
      case "Escape":
        setIsOpen(false);
        setSelectedIndex(-1);
        break;
    }
  };

  const handleSelect = (suggestion: {
    type: string;
    label: string;
    id?: string;
  }) => {
    if (suggestion.type === "document" && suggestion.id && onSelectDocument) {
      onSelectDocument(suggestion.id);
    } else {
      setQuery(suggestion.label);
      onSearch(suggestion.label);
    }
    setIsOpen(false);
    setSelectedIndex(-1);
  };

  const getIcon = (type: string) => {
    switch (type) {
      case "entity":
        return "🔖";
      case "document":
        return "📄";
      case "recent":
        return "🕐";
      case "topic":
        return "💡";
      default:
        return "🔍";
    }
  };

  return (
    <div style={{ position: "relative", width: "100%" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          backgroundColor: "#1f2937",
          borderRadius: "0.5rem",
          border: isOpen ? "1px solid #3b82f6" : "1px solid #374151",
          padding: "0.5rem 1rem",
          transition: "border-color 0.2s",
        }}
      >
        <span style={{ color: "#9ca3af", marginRight: "0.5rem" }}>🔍</span>
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setIsOpen(true);
            setSelectedIndex(-1);
          }}
          onFocus={() => setIsOpen(true)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          autoFocus={autoFocus}
          style={{
            flex: 1,
            backgroundColor: "transparent",
            border: "none",
            outline: "none",
            color: "#f9fafb",
            fontSize: "0.875rem",
          }}
        />
        {isLoading && (
          <span style={{ color: "#9ca3af", fontSize: "0.75rem" }}>...</span>
        )}
        {query && (
          <button
            onClick={() => {
              setQuery("");
              setSuggestions(null);
              inputRef.current?.focus();
            }}
            style={{
              background: "none",
              border: "none",
              color: "#9ca3af",
              cursor: "pointer",
              padding: "0.25rem",
            }}
          >
            ✕
          </button>
        )}
      </div>

      {/* Dropdown */}
      {isOpen && flatSuggestions.length > 0 && (
        <div
          ref={dropdownRef}
          style={{
            position: "absolute",
            top: "100%",
            left: 0,
            right: 0,
            marginTop: "0.25rem",
            backgroundColor: "#1f2937",
            border: "1px solid #374151",
            borderRadius: "0.5rem",
            maxHeight: "300px",
            overflowY: "auto",
            zIndex: 50,
            boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.3)",
          }}
        >
          {/* Entities */}
          {suggestions?.entities.length ? (
            <div style={{ padding: "0.5rem 0" }}>
              <div
                style={{
                  padding: "0.25rem 1rem",
                  fontSize: "0.625rem",
                  color: "#6b7280",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}
              >
                Entities
              </div>
              {suggestions.entities.slice(0, 5).map((entity, i) => {
                const idx = flatSuggestions.findIndex(
                  (s) => s.type === "entity" && s.label === entity.name
                );
                return (
                  <div
                    key={`entity-${i}`}
                    onClick={() =>
                      handleSelect({ type: "entity", label: entity.name })
                    }
                    style={{
                      padding: "0.5rem 1rem",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                      backgroundColor:
                        selectedIndex === idx ? "#374151" : "transparent",
                    }}
                  >
                    <span>{getIcon("entity")}</span>
                    <span style={{ color: "#f9fafb" }}>{entity.name}</span>
                    <span
                      style={{
                        marginLeft: "auto",
                        fontSize: "0.625rem",
                        color: "#6b7280",
                        backgroundColor: "#374151",
                        padding: "0.125rem 0.375rem",
                        borderRadius: "0.25rem",
                      }}
                    >
                      {entity.type}
                    </span>
                  </div>
                );
              })}
            </div>
          ) : null}

          {/* Documents */}
          {suggestions?.documents.length ? (
            <div
              style={{ padding: "0.5rem 0", borderTop: "1px solid #374151" }}
            >
              <div
                style={{
                  padding: "0.25rem 1rem",
                  fontSize: "0.625rem",
                  color: "#6b7280",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}
              >
                Documents
              </div>
              {suggestions.documents.slice(0, 5).map((doc, i) => {
                const idx = flatSuggestions.findIndex(
                  (s) => s.type === "document" && s.id === doc.id
                );
                return (
                  <div
                    key={`doc-${i}`}
                    onClick={() =>
                      handleSelect({
                        type: "document",
                        label: doc.title,
                        id: doc.id,
                      })
                    }
                    style={{
                      padding: "0.5rem 1rem",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                      backgroundColor:
                        selectedIndex === idx ? "#374151" : "transparent",
                    }}
                  >
                    <span>{getIcon("document")}</span>
                    <span style={{ color: "#f9fafb" }}>{doc.title}</span>
                  </div>
                );
              })}
            </div>
          ) : null}

          {/* Recent Queries */}
          {suggestions?.recent_queries.length ? (
            <div
              style={{ padding: "0.5rem 0", borderTop: "1px solid #374151" }}
            >
              <div
                style={{
                  padding: "0.25rem 1rem",
                  fontSize: "0.625rem",
                  color: "#6b7280",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                }}
              >
                Recent
              </div>
              {suggestions.recent_queries.slice(0, 3).map((q, i) => {
                const idx = flatSuggestions.findIndex(
                  (s) => s.type === "recent" && s.label === q
                );
                return (
                  <div
                    key={`recent-${i}`}
                    onClick={() => handleSelect({ type: "recent", label: q })}
                    style={{
                      padding: "0.5rem 1rem",
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                      backgroundColor:
                        selectedIndex === idx ? "#374151" : "transparent",
                    }}
                  >
                    <span>{getIcon("recent")}</span>
                    <span style={{ color: "#d1d5db" }}>{q}</span>
                  </div>
                );
              })}
            </div>
          ) : null}
        </div>
      )}
    </div>
  );
}

export default SearchBar;
