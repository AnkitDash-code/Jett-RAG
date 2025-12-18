"use client";

import { useCallback, useState, useRef } from "react";
import { fetchEventSource } from "@microsoft/fetch-event-source";
import { api, API_BASE_URL } from "@/lib/api";
import type { SourceCitation } from "@/types/api";

interface StreamCallbacks {
  onToken: (token: string) => void;
  onSources: (sources: SourceCitation[]) => void;
  onDone: (data: { conversation_id: string; message_id: string }) => void;
  onError: (error: string) => void;
}

export function useChatStream() {
  const [isStreaming, setIsStreaming] = useState(false);
  const abortControllerRef = useRef<AbortController | null>(null);

  const streamChat = useCallback(
    async (
      message: string,
      conversationId: string | null,
      callbacks: StreamCallbacks
    ) => {
      const accessToken = api.getAccessToken();
      if (!accessToken) {
        callbacks.onError("Not authenticated");
        return;
      }

      // Abort previous stream if any
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      const controller = new AbortController();
      abortControllerRef.current = controller;

      setIsStreaming(true);

      const params = new URLSearchParams({ message });
      if (conversationId) params.append("conversation_id", conversationId);

      try {
        console.log(`[StreamChat] Connecting to: ${API_BASE_URL}/chat/stream?${params}`);
        
        await fetchEventSource(`${API_BASE_URL}/chat/stream?${params}`, {
          method: "GET",
          headers: {
            Authorization: `Bearer ${accessToken}`,
            Accept: "text/event-stream",
          },
          signal: controller.signal,
          openWhenHidden: true, // Keep connection open even when tab is hidden
          onopen: async (response) => {
            console.log(`[StreamChat] Connection opened, status: ${response.status}, content-type: ${response.headers.get('content-type')}`);
            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }
            // Verify it's actually an SSE stream
            const contentType = response.headers.get('content-type');
            if (!contentType?.includes('text/event-stream')) {
              console.warn(`[StreamChat] Unexpected content-type: ${contentType}`);
            }
          },
          onmessage(event) {
            console.log(`[StreamChat] Event: "${event.event}", data length: ${event.data?.length}`);
            switch (event.event) {
              case "token":
                callbacks.onToken(event.data);
                break;
              case "sources":
                try {
                  callbacks.onSources(JSON.parse(event.data));
                } catch {
                  console.error("Failed to parse sources");
                }
                break;
              case "done":
                try {
                  callbacks.onDone(JSON.parse(event.data));
                } catch {
                  console.error("Failed to parse done event");
                }
                setIsStreaming(false);
                break;
              case "error":
                callbacks.onError(event.data);
                setIsStreaming(false);
                break;
            }
          },
          onerror(err) {
            console.error(`[StreamChat] Error:`, err);
            callbacks.onError(err.message || "Stream error");
            setIsStreaming(false);
            throw err; // Stop retrying
          },
          onclose() {
            console.log(`[StreamChat] Connection closed`);
            setIsStreaming(false);
          },
        });
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          callbacks.onError((error as Error).message || "Stream failed");
        }
        setIsStreaming(false);
      }
    },
    []
  );

  const stopStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  }, []);

  return { streamChat, stopStream, isStreaming };
}
