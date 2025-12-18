"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { toast } from "sonner";
import { Message, Document } from "@/types";
import { api } from "@/lib/api";
import { useChatStream } from "@/hooks/useChatStream";
import type {
  SourceCitation,
  DocumentResponse,
  ConversationSummary,
} from "@/types/api";

interface ChatMessage extends Omit<Message, "sources"> {
  sources?: SourceCitation[];
}

// Modal component for showing source content
function SourceModal({
  source,
  onClose,
}: {
  source: SourceCitation | null;
  onClose: () => void;
}) {
  if (!source) return null;

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: "rgba(0, 0, 0, 0.5)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 1000,
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: "#1f2937",
          borderRadius: "0.5rem",
          padding: "1.5rem",
          maxWidth: "600px",
          maxHeight: "80vh",
          overflow: "auto",
          margin: "1rem",
          border: "1px solid #374151",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            marginBottom: "1rem",
          }}
        >
          <div>
            <h3 style={{ margin: 0, color: "#f9fafb" }}>📄 {source.name}</h3>
            <div
              style={{
                fontSize: "0.875rem",
                color: "#9ca3af",
                marginTop: "0.25rem",
              }}
            >
              {source.page && <span>Page {source.page}</span>}
              {source.section && <span> • {source.section}</span>}
              <span
                style={{
                  marginLeft: "0.5rem",
                  backgroundColor:
                    source.relevance === "High"
                      ? "#065f46"
                      : source.relevance === "Medium"
                      ? "#92400e"
                      : "#374151",
                  color: "#fff",
                  padding: "0.125rem 0.5rem",
                  borderRadius: "9999px",
                  fontSize: "0.75rem",
                }}
              >
                {source.relevance} Relevance
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "none",
              color: "#9ca3af",
              cursor: "pointer",
              fontSize: "1.25rem",
              padding: "0.25rem",
            }}
          >
            ✕
          </button>
        </div>
        <div
          style={{
            backgroundColor: "#111827",
            padding: "1rem",
            borderRadius: "0.375rem",
            fontSize: "0.875rem",
            lineHeight: "1.6",
            color: "#d1d5db",
            whiteSpace: "pre-wrap",
            border: "1px solid #374151",
          }}
        >
          {source.snippet || "No content preview available."}
        </div>
      </div>
    </div>
  );
}

export default function Chat() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "1",
      role: "assistant",
      content:
        "Hello! I'm your GraphRAG assistant. Upload a document or ask me anything about your knowledge base.",
      timestamp: new Date(),
    },
  ]);
  const [documents, setDocuments] = useState<DocumentResponse[]>([]);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [selectedSource, setSelectedSource] = useState<SourceCitation | null>(
    null
  );
  const { streamChat, stopStream, isStreaming } = useChatStream();

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const streamingMessageRef = useRef<string>("");

  // Fetch documents and conversations on mount
  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        const response = await api.listDocuments();
        setDocuments(response.documents);
      } catch {
        // Silently fail - user may not have documents
      }
    };
    const fetchConversations = async () => {
      try {
        const response = await api.getChatHistory();
        setConversations(response.conversations);
      } catch {
        // Silently fail - user may not have conversations
      }
    };
    fetchDocuments();
    fetchConversations();
  }, []);

  // Track if we're currently polling (use ref to avoid re-render loops)
  const isPollingRef = useRef(false);

  // Auto-poll for document status updates when there are processing documents
  useEffect(() => {
    const hasProcessingDocs = documents.some(
      (doc) => !doc.status?.includes("indexed") && doc.status !== "error"
    );

    // Only start polling if not already polling and there are processing docs
    if (!hasProcessingDocs || isPollingRef.current) return;

    isPollingRef.current = true;

    const pollInterval = setInterval(async () => {
      try {
        const response = await api.listDocuments();

        // Check if any document is still processing
        const stillProcessing = response.documents.some(
          (doc) => !doc.status?.includes("indexed") && doc.status !== "error"
        );

        setDocuments(response.documents);

        if (!stillProcessing) {
          clearInterval(pollInterval);
          isPollingRef.current = false;
          toast.success("Document processing complete!");
        }
      } catch {
        // Silently fail
      }
    }, 3000); // Poll every 3 seconds

    return () => {
      clearInterval(pollInterval);
      isPollingRef.current = false;
    };
  }, [documents]); // Re-run when documents change

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Refresh document list
  const refreshDocuments = async () => {
    try {
      const response = await api.listDocuments();
      setDocuments(response.documents);
    } catch {
      // Silently fail
    }
  };

  // Delete document handler
  const handleDeleteDocument = async (docId: string, docName: string) => {
    if (
      !confirm(
        `Delete "${docName}"? This will remove the document and its index.`
      )
    ) {
      return;
    }
    try {
      await api.deleteDocument(docId);
      setDocuments((prev) => prev.filter((d) => d.id !== docId));
      toast.success(`"${docName}" deleted successfully`);
    } catch (error) {
      toast.error(
        `Failed to delete document: ${
          error instanceof Error ? error.message : "Unknown error"
        }`
      );
    }
  };

  // Document upload handler
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    setIsUploading(true);
    const file = acceptedFiles[0];

    try {
      const result = await api.uploadDocument(file);
      toast.success(`Uploaded ${result.filename}`);
      // Refresh the document list to get the full document info
      await refreshDocuments();
    } catch (error) {
      toast.error((error as Error).message || "Upload failed");
    } finally {
      setIsUploading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    maxSize: 10 * 1024 * 1024, // 10MB
    multiple: false,
  });

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    const query = input;
    setInput("");

    // Create placeholder for streaming response
    const assistantMessageId = (Date.now() + 1).toString();
    streamingMessageRef.current = "";

    setMessages((prev) => [
      ...prev,
      {
        id: assistantMessageId,
        role: "assistant",
        content: "",
        timestamp: new Date(),
      },
    ]);

    await streamChat(query, conversationId, {
      onToken: (token) => {
        streamingMessageRef.current += token;
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? { ...msg, content: streamingMessageRef.current }
              : msg
          )
        );
      },
      onSources: (sources) => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId ? { ...msg, sources } : msg
          )
        );
      },
      onDone: ({ conversation_id }) => {
        setConversationId(conversation_id);
        // Refresh conversations list
        api
          .getChatHistory()
          .then((res) => setConversations(res.conversations))
          .catch(() => {});
      },
      onError: (error) => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? { ...msg, content: `Error: ${error}` }
              : msg
          )
        );
        toast.error(error);
      },
    });
  };

  // Load a conversation from history
  const loadConversation = async (convId: string) => {
    try {
      const conversation = await api.getConversation(convId);
      setConversationId(convId);
      setMessages(
        conversation.messages.map((msg) => ({
          id: msg.id,
          role: msg.role as "user" | "assistant",
          content: msg.content,
          sources: msg.sources,
          timestamp: new Date(msg.created_at),
        }))
      );
      toast.success(`Loaded: ${conversation.title || "Conversation"}`);
    } catch (error) {
      toast.error("Failed to load conversation");
    }
  };

  // Start a new conversation
  const startNewConversation = () => {
    setConversationId(null);
    setMessages([
      {
        id: "1",
        role: "assistant",
        content:
          "Hello! I'm your GraphRAG assistant. Upload a document or ask me anything about your knowledge base.",
        timestamp: new Date(),
      },
    ]);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <main className="main-content chat-page">
      <header>
        <h2>Chat with your Knowledge Base</h2>
        <p>
          Ask questions about your uploaded documents securely and privately.
        </p>
      </header>

      <section className="chat-grid">
        {/* Left Panel: Upload & Sources */}
        <div className="panel upload-panel">
          <h3>Upload Documents</h3>
          <p
            style={{
              fontSize: "0.875rem",
              color: "#6b7280",
              marginBottom: "1rem",
            }}
          >
            Add PDFs to your knowledge graph.
          </p>

          <div
            {...getRootProps()}
            className={`drag-drop ${isDragActive ? "drag-active" : ""}`}
            style={{
              cursor: isUploading ? "wait" : "pointer",
              opacity: isUploading ? 0.6 : 1,
            }}
          >
            <input {...getInputProps()} />
            <i className="icon-lg">📄</i>
            <p style={{ fontWeight: 500 }}>
              {isUploading
                ? "Uploading..."
                : isDragActive
                ? "Drop the file here..."
                : "Drag & drop or click to upload"}
            </p>
            <small style={{ color: "#9ca3af" }}>PDF files up to 10MB</small>
          </div>

          <div style={{ marginTop: "auto" }}>
            <h3>Active Documents ({documents.length})</h3>
            <ul className="sources-list">
              {documents.length === 0 ? (
                <li style={{ color: "#9ca3af", fontStyle: "italic" }}>
                  No documents uploaded yet
                </li>
              ) : (
                documents.map((doc) => (
                  <li
                    key={doc.id}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      gap: "0.5rem",
                    }}
                  >
                    <span
                      style={{
                        fontSize: "0.875rem",
                        flex: 1,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {doc.filename}
                    </span>
                    <span
                      style={{
                        fontSize: "0.75rem",
                        backgroundColor: doc.status?.includes("indexed")
                          ? "#d1fae5"
                          : doc.status === "error"
                          ? "#fee2e2"
                          : "#fef3c7",
                        color: doc.status?.includes("indexed")
                          ? "#065f46"
                          : doc.status === "error"
                          ? "#dc2626"
                          : "#92400e",
                        padding: "0.125rem 0.5rem",
                        borderRadius: "9999px",
                        flexShrink: 0,
                      }}
                    >
                      {doc.status?.includes("indexed")
                        ? "Indexed"
                        : doc.status === "error"
                        ? "Error"
                        : "Processing"}
                    </span>
                    <button
                      onClick={() => handleDeleteDocument(doc.id, doc.filename)}
                      style={{
                        background: "none",
                        border: "none",
                        cursor: "pointer",
                        padding: "0.25rem",
                        fontSize: "1rem",
                        color: "#9ca3af",
                        flexShrink: 0,
                      }}
                      onMouseEnter={(e) =>
                        (e.currentTarget.style.color = "#ef4444")
                      }
                      onMouseLeave={(e) =>
                        (e.currentTarget.style.color = "#9ca3af")
                      }
                      title="Delete document"
                    >
                      🗑️
                    </button>
                  </li>
                ))
              )}
            </ul>
          </div>

          {/* Chat History */}
          <div style={{ marginTop: "1.5rem" }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: "0.5rem",
              }}
            >
              <h3 style={{ margin: 0 }}>Chat History</h3>
              <button
                onClick={startNewConversation}
                style={{
                  background: "none",
                  border: "1px solid #374151",
                  borderRadius: "0.25rem",
                  padding: "0.25rem 0.5rem",
                  fontSize: "0.75rem",
                  cursor: "pointer",
                  color: "#9ca3af",
                }}
                title="Start new chat"
              >
                + New
              </button>
            </div>
            <ul
              className="sources-list"
              style={{ maxHeight: "200px", overflowY: "auto" }}
            >
              {conversations.length === 0 ? (
                <li style={{ color: "#9ca3af", fontStyle: "italic" }}>
                  No past conversations
                </li>
              ) : (
                conversations.map((conv) => (
                  <li
                    key={conv.id}
                    onClick={() => loadConversation(conv.id)}
                    style={{
                      cursor: "pointer",
                      padding: "0.5rem",
                      borderRadius: "0.25rem",
                      backgroundColor:
                        conversationId === conv.id
                          ? "rgba(59, 130, 246, 0.2)"
                          : "transparent",
                      borderLeft:
                        conversationId === conv.id
                          ? "3px solid #3b82f6"
                          : "3px solid transparent",
                    }}
                    onMouseEnter={(e) => {
                      if (conversationId !== conv.id) {
                        e.currentTarget.style.backgroundColor =
                          "rgba(255,255,255,0.05)";
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (conversationId !== conv.id) {
                        e.currentTarget.style.backgroundColor = "transparent";
                      }
                    }}
                  >
                    <div
                      style={{
                        fontSize: "0.875rem",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      💬 {conv.title || `Chat ${conv.id.slice(0, 8)}...`}
                    </div>
                    <div
                      style={{
                        fontSize: "0.7rem",
                        color: "#6b7280",
                        marginTop: "0.125rem",
                      }}
                    >
                      {conv.message_count} messages •{" "}
                      {new Date(conv.created_at).toLocaleDateString()}
                    </div>
                  </li>
                ))
              )}
            </ul>
          </div>
        </div>

        {/* Right Panel: Chat Interface */}
        <div className="panel question-panel" style={{ position: "relative" }}>
          {/* Messages Area */}
          <div
            className="answer-box"
            style={{ marginBottom: "1rem", paddingRight: "0.5rem" }}
          >
            {messages.map((msg) => (
              <div key={msg.id} className={`message-container ${msg.role}`}>
                <div className="message-bubble">
                  <div className="message-header">
                    <i className="icon">{msg.role === "user" ? "👤" : "🤖"}</i>
                    <span>{msg.role === "user" ? "You" : "Assistant"}</span>
                  </div>
                  <p style={{ whiteSpace: "pre-wrap" }}>{msg.content}</p>

                  {/* Sources for Assistant Messages */}
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="message-sources">
                      <h4>Sources:</h4>
                      <ul>
                        {msg.sources.map((source, idx) => (
                          <li
                            key={idx}
                            onClick={() => setSelectedSource(source)}
                            style={{
                              cursor: "pointer",
                              padding: "0.5rem",
                              borderRadius: "0.25rem",
                              transition: "background-color 0.2s",
                            }}
                            onMouseEnter={(e) =>
                              (e.currentTarget.style.backgroundColor =
                                "rgba(255,255,255,0.1)")
                            }
                            onMouseLeave={(e) =>
                              (e.currentTarget.style.backgroundColor =
                                "transparent")
                            }
                            title="Click to view source content"
                          >
                            <span>📄 {source.name}</span>
                            {source.page && (
                              <span
                                style={{ fontSize: "0.75rem", opacity: 0.75 }}
                              >
                                {" "}
                                (Page {source.page})
                              </span>
                            )}
                            <span
                              className="relevance-badge"
                              style={{
                                marginLeft: "0.5rem",
                                fontSize: "0.75rem",
                                backgroundColor:
                                  source.relevance === "High"
                                    ? "#d1fae5"
                                    : source.relevance === "Medium"
                                    ? "#fef3c7"
                                    : "#f3f4f6",
                                color:
                                  source.relevance === "High"
                                    ? "#065f46"
                                    : source.relevance === "Medium"
                                    ? "#92400e"
                                    : "#374151",
                                padding: "0.125rem 0.5rem",
                                borderRadius: "9999px",
                              }}
                            >
                              {source.relevance}
                            </span>
                            <span
                              style={{
                                marginLeft: "0.5rem",
                                fontSize: "0.75rem",
                                color: "#60a5fa",
                              }}
                            >
                              🔍 View
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {isStreaming && messages[messages.length - 1]?.content === "" && (
              <div className="message-container assistant">
                <div
                  className="message-bubble"
                  style={{ padding: "0.5rem 1rem" }}
                >
                  <div className="loading-dots">
                    <div className="loading-dot"></div>
                    <div className="loading-dot"></div>
                    <div className="loading-dot"></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div
            className="query-box"
            style={{
              marginTop: "auto",
              paddingTop: "1rem",
              borderTop: "1px solid var(--color-grey-border)",
            }}
          >
            <div style={{ position: "relative" }}>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question about your documents..."
                style={{ height: "100px" }}
              />
              <button
                onClick={isStreaming ? stopStream : handleSend}
                disabled={!input.trim() && !isStreaming}
                className="btn btn-dark send-btn"
                style={{
                  position: "absolute",
                  bottom: "1rem",
                  right: "1rem",
                  opacity: !input.trim() && !isStreaming ? 0.5 : 1,
                }}
              >
                {isStreaming ? "Stop ■" : "Send ↵"}
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* Source Content Modal */}
      <SourceModal
        source={selectedSource}
        onClose={() => setSelectedSource(null)}
      />
    </main>
  );
}
