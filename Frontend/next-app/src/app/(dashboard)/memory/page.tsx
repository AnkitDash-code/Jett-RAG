"use client";

import { useState, useEffect, useCallback } from "react";
import { api, API_BASE_URL } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";

interface EpisodicMemory {
  id: string;
  event_type: string;
  content: string;
  importance: number;
  created_at: string;
  session_id?: string;
}

interface SemanticMemory {
  id: string;
  concept: string;
  summary: string;
  strength: number;
  last_accessed: string;
  related_concepts?: string[];
}

type MemoryTab = "episodic" | "semantic";

export default function MemoryExplorerPage() {
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState<MemoryTab>("episodic");
  const [episodicMemories, setEpisodicMemories] = useState<EpisodicMemory[]>(
    []
  );
  const [semanticMemories, setSemanticMemories] = useState<SemanticMemory[]>(
    []
  );
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");

  const fetchMemories = useCallback(async () => {
    try {
      setIsLoading(true);

      // Fetch episodic memories
      const episodicRes = await fetch(`${API_BASE_URL}/memory/episodic`, {
        headers: {
          Authorization: `Bearer ${api.getAccessToken()}`,
        },
      });

      if (episodicRes.ok) {
        const data = await episodicRes.json();
        setEpisodicMemories(data.memories || []);
      }

      // Fetch semantic memories
      const semanticRes = await fetch(`${API_BASE_URL}/memory/semantic`, {
        headers: {
          Authorization: `Bearer ${api.getAccessToken()}`,
        },
      });

      if (semanticRes.ok) {
        const data = await semanticRes.json();
        setSemanticMemories(data.concepts || []);
      }
    } catch (err) {
      console.error("Memory fetch error:", err);
      // Demo data
      setEpisodicMemories([
        {
          id: "1",
          event_type: "query",
          content: "Asked about document indexing process",
          importance: 0.8,
          created_at: new Date().toISOString(),
          session_id: "session-1",
        },
        {
          id: "2",
          event_type: "upload",
          content: "Uploaded technical_specification.pdf",
          importance: 0.9,
          created_at: new Date(Date.now() - 3600000).toISOString(),
          session_id: "session-1",
        },
        {
          id: "3",
          event_type: "feedback",
          content: "Provided positive feedback on response",
          importance: 0.6,
          created_at: new Date(Date.now() - 86400000).toISOString(),
          session_id: "session-2",
        },
      ]);

      setSemanticMemories([
        {
          id: "1",
          concept: "Document Processing",
          summary:
            "User frequently works with PDF documents and asks about parsing, chunking, and indexing workflows.",
          strength: 0.85,
          last_accessed: new Date().toISOString(),
          related_concepts: [
            "PDF Parsing",
            "Text Chunking",
            "Vector Embeddings",
          ],
        },
        {
          id: "2",
          concept: "RAG Architecture",
          summary:
            "User has explored retrieval-augmented generation concepts including hybrid search and reranking.",
          strength: 0.72,
          last_accessed: new Date(Date.now() - 86400000).toISOString(),
          related_concepts: ["Hybrid Search", "Reranking", "LLM Integration"],
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMemories();
  }, [fetchMemories]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/memory/recall`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${api.getAccessToken()}`,
        },
        body: JSON.stringify({ query: searchQuery }),
      });

      if (response.ok) {
        const data = await response.json();
        if (data.episodic) setEpisodicMemories(data.episodic);
        if (data.semantic) setSemanticMemories(data.semantic);
        toast.success("Found matching memories");
      }
    } catch (err) {
      toast.error("Search failed");
    } finally {
      setIsLoading(false);
    }
  };

  const getImportanceColor = (importance: number) => {
    if (importance >= 0.8) return "#10b981";
    if (importance >= 0.5) return "#f59e0b";
    return "#6b7280";
  };

  const getStrengthWidth = (strength: number) => {
    return `${Math.round(strength * 100)}%`;
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return (
      date.toLocaleDateString() +
      " " +
      date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    );
  };

  const getEventIcon = (type: string) => {
    switch (type) {
      case "query":
        return "💬";
      case "upload":
        return "📤";
      case "feedback":
        return "👍";
      case "action":
        return "⚡";
      default:
        return "📝";
    }
  };

  return (
    <main className="main-content memory-page">
      <header>
        <h2>Memory Explorer</h2>
        <p>View and search your episodic and semantic memories.</p>
      </header>

      {/* Search */}
      <div
        style={{
          display: "flex",
          gap: "0.5rem",
          marginBottom: "1.5rem",
        }}
      >
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          placeholder="Search across memories..."
          style={{
            flex: 1,
            padding: "0.75rem 1rem",
            borderRadius: "0.5rem",
            border: "1px solid #374151",
            backgroundColor: "#1f2937",
            color: "#f9fafb",
            fontSize: "0.875rem",
          }}
        />
        <button
          onClick={handleSearch}
          style={{
            padding: "0.75rem 1.5rem",
            borderRadius: "0.5rem",
            backgroundColor: "#3b82f6",
            color: "white",
            border: "none",
            cursor: "pointer",
            fontSize: "0.875rem",
          }}
        >
          Search
        </button>
      </div>

      {/* Tabs */}
      <div
        style={{
          display: "flex",
          gap: "0.5rem",
          marginBottom: "1rem",
          borderBottom: "1px solid #374151",
          paddingBottom: "0.5rem",
        }}
      >
        <button
          onClick={() => setActiveTab("episodic")}
          style={{
            padding: "0.5rem 1rem",
            borderRadius: "0.375rem",
            backgroundColor:
              activeTab === "episodic" ? "#3b82f6" : "transparent",
            color: activeTab === "episodic" ? "white" : "#9ca3af",
            border: "none",
            cursor: "pointer",
            fontSize: "0.875rem",
            fontWeight: 500,
          }}
        >
          📅 Episodic ({episodicMemories.length})
        </button>
        <button
          onClick={() => setActiveTab("semantic")}
          style={{
            padding: "0.5rem 1rem",
            borderRadius: "0.375rem",
            backgroundColor:
              activeTab === "semantic" ? "#3b82f6" : "transparent",
            color: activeTab === "semantic" ? "white" : "#9ca3af",
            border: "none",
            cursor: "pointer",
            fontSize: "0.875rem",
            fontWeight: 500,
          }}
        >
          🧠 Semantic ({semanticMemories.length})
        </button>
      </div>

      {/* Content */}
      {isLoading ? (
        <div className="panel" style={{ textAlign: "center", padding: "2rem" }}>
          Loading memories...
        </div>
      ) : activeTab === "episodic" ? (
        <div className="episodic-timeline">
          {episodicMemories.length === 0 ? (
            <div
              className="panel"
              style={{ textAlign: "center", padding: "2rem" }}
            >
              No episodic memories found.
            </div>
          ) : (
            <div
              style={{
                position: "relative",
                paddingLeft: "2rem",
                borderLeft: "2px solid #374151",
              }}
            >
              {episodicMemories.map((memory, index) => (
                <div
                  key={memory.id}
                  style={{
                    position: "relative",
                    marginBottom: "1.5rem",
                  }}
                >
                  {/* Timeline dot */}
                  <div
                    style={{
                      position: "absolute",
                      left: "-2.5rem",
                      width: "1rem",
                      height: "1rem",
                      borderRadius: "50%",
                      backgroundColor: getImportanceColor(memory.importance),
                      border: "2px solid #111827",
                    }}
                  />

                  {/* Memory card */}
                  <div
                    className="panel"
                    style={{
                      padding: "1rem",
                      marginLeft: "0.5rem",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "flex-start",
                        marginBottom: "0.5rem",
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: "0.5rem",
                        }}
                      >
                        <span style={{ fontSize: "1.25rem" }}>
                          {getEventIcon(memory.event_type)}
                        </span>
                        <span
                          style={{
                            fontSize: "0.75rem",
                            color: "#9ca3af",
                            textTransform: "capitalize",
                          }}
                        >
                          {memory.event_type}
                        </span>
                      </div>
                      <span style={{ fontSize: "0.75rem", color: "#6b7280" }}>
                        {formatDate(memory.created_at)}
                      </span>
                    </div>

                    <p
                      style={{
                        margin: 0,
                        color: "#f9fafb",
                        fontSize: "0.875rem",
                      }}
                    >
                      {memory.content}
                    </p>

                    <div
                      style={{
                        marginTop: "0.5rem",
                        display: "flex",
                        alignItems: "center",
                        gap: "0.5rem",
                      }}
                    >
                      <span style={{ fontSize: "0.75rem", color: "#6b7280" }}>
                        Importance:
                      </span>
                      <div
                        style={{
                          flex: 1,
                          maxWidth: "100px",
                          height: "4px",
                          backgroundColor: "#374151",
                          borderRadius: "2px",
                        }}
                      >
                        <div
                          style={{
                            width: getStrengthWidth(memory.importance),
                            height: "100%",
                            backgroundColor: getImportanceColor(
                              memory.importance
                            ),
                            borderRadius: "2px",
                          }}
                        />
                      </div>
                      <span style={{ fontSize: "0.75rem", color: "#9ca3af" }}>
                        {Math.round(memory.importance * 100)}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="semantic-concepts">
          {semanticMemories.length === 0 ? (
            <div
              className="panel"
              style={{ textAlign: "center", padding: "2rem" }}
            >
              No semantic memories found.
            </div>
          ) : (
            <div
              style={{ display: "flex", flexDirection: "column", gap: "1rem" }}
            >
              {semanticMemories.map((memory) => (
                <div
                  key={memory.id}
                  className="panel"
                  style={{ padding: "1.25rem" }}
                >
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "flex-start",
                      marginBottom: "0.75rem",
                    }}
                  >
                    <h4
                      style={{
                        margin: 0,
                        color: "#f9fafb",
                        display: "flex",
                        alignItems: "center",
                        gap: "0.5rem",
                      }}
                    >
                      🧠 {memory.concept}
                    </h4>
                    <span style={{ fontSize: "0.75rem", color: "#6b7280" }}>
                      Last accessed: {formatDate(memory.last_accessed)}
                    </span>
                  </div>

                  <p
                    style={{
                      margin: "0 0 1rem 0",
                      color: "#d1d5db",
                      fontSize: "0.875rem",
                      lineHeight: 1.6,
                    }}
                  >
                    {memory.summary}
                  </p>

                  {/* Strength bar */}
                  <div
                    style={{
                      marginBottom: "0.75rem",
                      display: "flex",
                      alignItems: "center",
                      gap: "0.5rem",
                    }}
                  >
                    <span style={{ fontSize: "0.75rem", color: "#6b7280" }}>
                      Strength:
                    </span>
                    <div
                      style={{
                        flex: 1,
                        maxWidth: "200px",
                        height: "6px",
                        backgroundColor: "#374151",
                        borderRadius: "3px",
                      }}
                    >
                      <div
                        style={{
                          width: getStrengthWidth(memory.strength),
                          height: "100%",
                          backgroundColor: "#3b82f6",
                          borderRadius: "3px",
                        }}
                      />
                    </div>
                    <span style={{ fontSize: "0.75rem", color: "#9ca3af" }}>
                      {Math.round(memory.strength * 100)}%
                    </span>
                  </div>

                  {/* Related concepts */}
                  {memory.related_concepts &&
                    memory.related_concepts.length > 0 && (
                      <div
                        style={{
                          display: "flex",
                          flexWrap: "wrap",
                          gap: "0.5rem",
                        }}
                      >
                        {memory.related_concepts.map((concept, i) => (
                          <span
                            key={i}
                            style={{
                              padding: "0.25rem 0.75rem",
                              borderRadius: "9999px",
                              backgroundColor: "#1f2937",
                              color: "#60a5fa",
                              fontSize: "0.75rem",
                              border: "1px solid #374151",
                            }}
                          >
                            {concept}
                          </span>
                        ))}
                      </div>
                    )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </main>
  );
}
