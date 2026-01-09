"use client";

import { useState, useRef } from "react";
import { toast } from "sonner";
import { api } from "@/lib/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp?: Date;
  sources?: Array<{
    name: string;
    page?: number;
    section?: string;
    snippet?: string;
    relevance?: string;
  }>;
}

interface ConversationExporterProps {
  conversationId?: string;
  messages: Message[];
  onImport?: (messages: Message[]) => void;
}

export function ConversationExporter({
  conversationId,
  messages,
  onImport,
}: ConversationExporterProps) {
  const [isExporting, setIsExporting] = useState(false);
  const [showMenu, setShowMenu] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const exportAsJson = async () => {
    setIsExporting(true);
    try {
      const exportData = {
        version: "1.0",
        exported_at: new Date().toISOString(),
        conversation_id: conversationId,
        message_count: messages.length,
        messages: messages.map((msg) => ({
          id: msg.id,
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp?.toISOString(),
          sources: msg.sources,
        })),
      };

      const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `conversation-${
        conversationId || "export"
      }-${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      toast.success("Conversation exported as JSON");
    } catch (err) {
      toast.error("Failed to export conversation");
    } finally {
      setIsExporting(false);
      setShowMenu(false);
    }
  };

  const exportAsMarkdown = async () => {
    setIsExporting(true);
    try {
      let markdown = `# Conversation Export\n\n`;
      markdown += `**Exported:** ${new Date().toLocaleString()}\n`;
      markdown += `**Messages:** ${messages.length}\n\n---\n\n`;

      for (const msg of messages) {
        const role = msg.role === "user" ? "👤 **You**" : "🤖 **Assistant**";
        const time = msg.timestamp
          ? ` _(${new Date(msg.timestamp).toLocaleString()})_`
          : "";

        markdown += `## ${role}${time}\n\n`;
        markdown += `${msg.content}\n\n`;

        if (msg.sources && msg.sources.length > 0) {
          markdown += `### Sources\n\n`;
          msg.sources.forEach((source, idx) => {
            markdown += `${idx + 1}. **${source.name}**`;
            if (source.page) markdown += ` (Page ${source.page})`;
            if (source.relevance)
              markdown += ` - _${source.relevance} relevance_`;
            markdown += `\n`;
            if (source.snippet) {
              markdown += `   > ${source.snippet.substring(0, 200)}...\n`;
            }
          });
          markdown += `\n`;
        }

        markdown += `---\n\n`;
      }

      const blob = new Blob([markdown], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `conversation-${
        conversationId || "export"
      }-${Date.now()}.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      toast.success("Conversation exported as Markdown");
    } catch (err) {
      toast.error("Failed to export conversation");
    } finally {
      setIsExporting(false);
      setShowMenu(false);
    }
  };

  const exportAsPlainText = async () => {
    setIsExporting(true);
    try {
      let text = `CONVERSATION EXPORT\n`;
      text += `Exported: ${new Date().toLocaleString()}\n`;
      text += `Messages: ${messages.length}\n`;
      text += `${"=".repeat(50)}\n\n`;

      for (const msg of messages) {
        const role = msg.role === "user" ? "YOU" : "ASSISTANT";
        text += `[${role}]\n`;
        text += `${msg.content}\n\n`;

        if (msg.sources && msg.sources.length > 0) {
          text += `Sources:\n`;
          msg.sources.forEach((source, idx) => {
            text += `  ${idx + 1}. ${source.name}`;
            if (source.page) text += ` (Page ${source.page})`;
            text += `\n`;
          });
          text += `\n`;
        }

        text += `${"-".repeat(50)}\n\n`;
      }

      const blob = new Blob([text], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `conversation-${
        conversationId || "export"
      }-${Date.now()}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      toast.success("Conversation exported as plain text");
    } catch (err) {
      toast.error("Failed to export conversation");
    } finally {
      setIsExporting(false);
      setShowMenu(false);
    }
  };

  const handleImport = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const data = JSON.parse(text);

      if (!data.messages || !Array.isArray(data.messages)) {
        throw new Error("Invalid conversation format");
      }

      const importedMessages: Message[] = data.messages.map((msg: any) => ({
        id: msg.id || `imported-${Date.now()}-${Math.random()}`,
        role: msg.role,
        content: msg.content,
        timestamp: msg.timestamp ? new Date(msg.timestamp) : new Date(),
        sources: msg.sources,
      }));

      onImport?.(importedMessages);
      toast.success(`Imported ${importedMessages.length} messages`);
    } catch (err) {
      toast.error(
        "Failed to import conversation. Please check the file format."
      );
    }

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    setShowMenu(false);
  };

  const copyToClipboard = async () => {
    try {
      const text = messages
        .map(
          (msg) =>
            `${msg.role === "user" ? "You" : "Assistant"}: ${msg.content}`
        )
        .join("\n\n");

      await navigator.clipboard.writeText(text);
      toast.success("Conversation copied to clipboard");
    } catch (err) {
      toast.error("Failed to copy to clipboard");
    }
    setShowMenu(false);
  };

  return (
    <div style={{ position: "relative" }}>
      <button
        onClick={() => setShowMenu(!showMenu)}
        disabled={isExporting || messages.length === 0}
        style={{
          padding: "0.5rem 1rem",
          borderRadius: "0.375rem",
          backgroundColor: isExporting ? "#374151" : "#3b82f6",
          color: "#fff",
          border: "none",
          cursor: messages.length === 0 ? "not-allowed" : "pointer",
          fontSize: "0.875rem",
          display: "flex",
          alignItems: "center",
          gap: "0.5rem",
          opacity: messages.length === 0 ? 0.5 : 1,
        }}
        title={
          messages.length === 0
            ? "No messages to export"
            : "Export conversation"
        }
      >
        {isExporting ? <>⏳ Exporting...</> : <>📤 Export</>}
      </button>

      {showMenu && (
        <>
          {/* Backdrop */}
          <div
            style={{
              position: "fixed",
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              zIndex: 999,
            }}
            onClick={() => setShowMenu(false)}
          />

          {/* Menu */}
          <div
            style={{
              position: "absolute",
              top: "100%",
              right: 0,
              marginTop: "0.5rem",
              backgroundColor: "#1f2937",
              border: "1px solid #374151",
              borderRadius: "0.5rem",
              padding: "0.5rem",
              minWidth: "180px",
              zIndex: 1000,
              boxShadow: "0 10px 25px rgba(0, 0, 0, 0.3)",
            }}
          >
            <button
              onClick={exportAsJson}
              style={{
                width: "100%",
                padding: "0.5rem 0.75rem",
                textAlign: "left",
                backgroundColor: "transparent",
                border: "none",
                color: "#f9fafb",
                cursor: "pointer",
                borderRadius: "0.25rem",
                fontSize: "0.875rem",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.backgroundColor = "#374151")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.backgroundColor = "transparent")
              }
            >
              📄 Export as JSON
            </button>
            <button
              onClick={exportAsMarkdown}
              style={{
                width: "100%",
                padding: "0.5rem 0.75rem",
                textAlign: "left",
                backgroundColor: "transparent",
                border: "none",
                color: "#f9fafb",
                cursor: "pointer",
                borderRadius: "0.25rem",
                fontSize: "0.875rem",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.backgroundColor = "#374151")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.backgroundColor = "transparent")
              }
            >
              📝 Export as Markdown
            </button>
            <button
              onClick={exportAsPlainText}
              style={{
                width: "100%",
                padding: "0.5rem 0.75rem",
                textAlign: "left",
                backgroundColor: "transparent",
                border: "none",
                color: "#f9fafb",
                cursor: "pointer",
                borderRadius: "0.25rem",
                fontSize: "0.875rem",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.backgroundColor = "#374151")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.backgroundColor = "transparent")
              }
            >
              📃 Export as Plain Text
            </button>
            <button
              onClick={copyToClipboard}
              style={{
                width: "100%",
                padding: "0.5rem 0.75rem",
                textAlign: "left",
                backgroundColor: "transparent",
                border: "none",
                color: "#f9fafb",
                cursor: "pointer",
                borderRadius: "0.25rem",
                fontSize: "0.875rem",
              }}
              onMouseEnter={(e) =>
                (e.currentTarget.style.backgroundColor = "#374151")
              }
              onMouseLeave={(e) =>
                (e.currentTarget.style.backgroundColor = "transparent")
              }
            >
              📋 Copy to Clipboard
            </button>

            {onImport && (
              <>
                <div
                  style={{
                    borderTop: "1px solid #374151",
                    margin: "0.5rem 0",
                  }}
                />
                <label
                  style={{
                    display: "block",
                    width: "100%",
                    padding: "0.5rem 0.75rem",
                    textAlign: "left",
                    backgroundColor: "transparent",
                    color: "#f9fafb",
                    cursor: "pointer",
                    borderRadius: "0.25rem",
                    fontSize: "0.875rem",
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.backgroundColor = "#374151")
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.backgroundColor = "transparent")
                  }
                >
                  📥 Import from JSON
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".json"
                    onChange={handleImport}
                    style={{ display: "none" }}
                  />
                </label>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default ConversationExporter;
