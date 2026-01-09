"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import dynamic from "next/dynamic";
import { api } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

// Dynamically import force-graph to avoid SSR issues
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
  loading: () => (
    <div
      style={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        height: "500px",
      }}
    >
      Loading graph...
    </div>
  ),
});

interface GraphNode {
  id: string;
  name: string;
  type: string;
  community?: number;
  val?: number;
}

interface GraphLink {
  source: string;
  target: string;
  type: string;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

// Color palette for communities
const COMMUNITY_COLORS = [
  "#3b82f6", // blue
  "#10b981", // green
  "#f59e0b", // amber
  "#ef4444", // red
  "#8b5cf6", // purple
  "#06b6d4", // cyan
  "#f97316", // orange
  "#ec4899", // pink
  "#84cc16", // lime
  "#6366f1", // indigo
];

// Colors for entity types
const TYPE_COLORS: Record<string, string> = {
  PERSON: "#3b82f6",
  ORGANIZATION: "#10b981",
  LOCATION: "#f59e0b",
  CONCEPT: "#8b5cf6",
  EVENT: "#ef4444",
  TECHNOLOGY: "#06b6d4",
  DOCUMENT: "#ec4899",
  DEFAULT: "#6b7280",
};

export default function GraphVisualizationPage() {
  const { user } = useAuth();
  const router = useRouter();
  const graphRef = useRef<any>();

  const [graphData, setGraphData] = useState<GraphData>({
    nodes: [],
    links: [],
  });
  const [isLoading, setIsLoading] = useState(true);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [colorBy, setColorBy] = useState<"community" | "type">("type");
  const [searchQuery, setSearchQuery] = useState("");
  const [highlightNodes, setHighlightNodes] = useState<Set<string>>(new Set());
  const [stats, setStats] = useState({
    entities: 0,
    relationships: 0,
    communities: 0,
  });

  // Redirect non-admins
  useEffect(() => {
    if (user && user.role !== "admin") {
      router.push("/dashboard");
      toast.error("Access denied. Admin privileges required.");
    }
  }, [user, router]);

  // Fetch graph data
  const fetchGraphData = useCallback(async () => {
    try {
      setIsLoading(true);
      const response = await fetch(
        `${
          process.env.NEXT_PUBLIC_API_URL || "http://localhost:8081/v1"
        }/admin/graph/visualization?limit=100&include_orphans=false`,
        {
          headers: {
            Authorization: `Bearer ${api.getAccessToken()}`,
          },
        }
      );

      if (!response.ok) {
        throw new Error("Failed to fetch graph data");
      }

      const data = await response.json();

      // Transform to force-graph format
      const nodes: GraphNode[] = (data.entities || []).map((e: any) => ({
        id: e.id || e.name,
        name: e.name,
        type: e.type || "CONCEPT",
        community: e.community_id
          ? parseInt(e.community_id.split("-")[0], 16) % 10
          : undefined,
        val: Math.max(1, e.connection_count || 1),
      }));

      const links: GraphLink[] = (data.relationships || []).map((r: any) => ({
        source: r.source_id || r.source,
        target: r.target_id || r.target,
        type: r.relationship_type || r.type || "RELATED",
      }));

      setGraphData({ nodes, links });
      setStats({
        entities: data.stats?.total_entities || nodes.length,
        relationships: data.stats?.total_relationships || links.length,
        communities:
          data.stats?.communities ||
          new Set(nodes.map((n) => n.community).filter(Boolean)).size,
      });

      if (nodes.length > 0) {
        toast.success(
          `Loaded ${nodes.length} entities, ${links.length} relationships`
        );
      } else {
        toast.info(
          "No graph data available. Upload documents to build the knowledge graph."
        );
      }
    } catch (err) {
      console.error("Graph fetch error:", err);
      // Use mock data for demo
      setGraphData({
        nodes: [
          {
            id: "1",
            name: "Document A",
            type: "DOCUMENT",
            community: 0,
            val: 3,
          },
          { id: "2", name: "Concept X", type: "CONCEPT", community: 0, val: 2 },
          { id: "3", name: "Person Y", type: "PERSON", community: 1, val: 2 },
          {
            id: "4",
            name: "Organization Z",
            type: "ORGANIZATION",
            community: 1,
            val: 1,
          },
          {
            id: "5",
            name: "Location W",
            type: "LOCATION",
            community: 2,
            val: 1,
          },
        ],
        links: [
          { source: "1", target: "2", type: "CONTAINS" },
          { source: "2", target: "3", type: "RELATED" },
          { source: "3", target: "4", type: "WORKS_AT" },
          { source: "4", target: "5", type: "LOCATED_IN" },
        ],
      });
      setStats({ entities: 5, relationships: 4, communities: 3 });
      toast.info("Using demo graph data");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchGraphData();
  }, [fetchGraphData]);

  // Search/highlight nodes
  useEffect(() => {
    if (!searchQuery.trim()) {
      setHighlightNodes(new Set());
      return;
    }

    const query = searchQuery.toLowerCase();
    const matching = graphData.nodes
      .filter((n) => n.name.toLowerCase().includes(query))
      .map((n) => n.id);
    setHighlightNodes(new Set(matching));
  }, [searchQuery, graphData.nodes]);

  // Get node color
  const getNodeColor = (node: GraphNode) => {
    if (highlightNodes.size > 0 && !highlightNodes.has(node.id)) {
      return "#374151"; // dimmed
    }

    if (colorBy === "community" && node.community !== undefined) {
      return COMMUNITY_COLORS[node.community % COMMUNITY_COLORS.length];
    }

    return TYPE_COLORS[node.type] || TYPE_COLORS.DEFAULT;
  };

  if (user?.role !== "admin") {
    return null;
  }

  return (
    <main className="main-content graph-page">
      <header>
        <h2>Knowledge Graph</h2>
        <p>Visualize entity relationships and communities.</p>
      </header>

      {/* Controls */}
      <div
        className="graph-controls"
        style={{
          display: "flex",
          gap: "1rem",
          marginBottom: "1rem",
          flexWrap: "wrap",
          alignItems: "center",
        }}
      >
        <input
          type="text"
          placeholder="Search entities..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{
            padding: "0.5rem 1rem",
            borderRadius: "0.375rem",
            border: "1px solid #374151",
            backgroundColor: "#1f2937",
            color: "#f9fafb",
            minWidth: "200px",
          }}
        />

        <select
          value={colorBy}
          onChange={(e) => setColorBy(e.target.value as "community" | "type")}
          style={{
            padding: "0.5rem 1rem",
            borderRadius: "0.375rem",
            border: "1px solid #374151",
            backgroundColor: "#1f2937",
            color: "#f9fafb",
          }}
        >
          <option value="type">Color by Type</option>
          <option value="community">Color by Community</option>
        </select>

        <button
          onClick={() => graphRef.current?.zoomToFit(400)}
          style={{
            padding: "0.5rem 1rem",
            borderRadius: "0.375rem",
            backgroundColor: "#3b82f6",
            color: "white",
            border: "none",
            cursor: "pointer",
          }}
        >
          Fit View
        </button>

        <button
          onClick={fetchGraphData}
          style={{
            padding: "0.5rem 1rem",
            borderRadius: "0.375rem",
            backgroundColor: "#374151",
            color: "white",
            border: "none",
            cursor: "pointer",
          }}
        >
          Refresh
        </button>
      </div>

      {/* Stats */}
      <div
        className="graph-stats"
        style={{
          display: "flex",
          gap: "1.5rem",
          marginBottom: "1rem",
        }}
      >
        <span>
          <strong>{stats.entities}</strong> Entities
        </span>
        <span>
          <strong>{stats.relationships}</strong> Relationships
        </span>
        <span>
          <strong>{stats.communities}</strong> Communities
        </span>
      </div>

      {/* Graph Container */}
      <div
        className="graph-container"
        style={{
          backgroundColor: "#111827",
          borderRadius: "0.5rem",
          border: "1px solid #374151",
          overflow: "hidden",
          position: "relative",
        }}
      >
        {isLoading ? (
          <div
            style={{
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              height: "500px",
            }}
          >
            Loading graph...
          </div>
        ) : (
          <ForceGraph2D
            ref={graphRef}
            graphData={graphData}
            width={800}
            height={500}
            nodeLabel={(node: any) => `${node.name} (${node.type})`}
            nodeColor={(node: any) => getNodeColor(node)}
            nodeRelSize={6}
            nodeVal={(node: any) => node.val || 1}
            linkColor={() => "#4b5563"}
            linkWidth={1.5}
            linkDirectionalParticles={2}
            linkDirectionalParticleWidth={2}
            onNodeClick={(node: any) => setSelectedNode(node)}
            cooldownTicks={100}
            onEngineStop={() => graphRef.current?.zoomToFit(400)}
          />
        )}
      </div>

      {/* Legend */}
      <div
        className="graph-legend"
        style={{
          display: "flex",
          gap: "1rem",
          marginTop: "1rem",
          flexWrap: "wrap",
        }}
      >
        {colorBy === "type"
          ? Object.entries(TYPE_COLORS).map(([type, color]) => (
              <div
                key={type}
                style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
              >
                <div
                  style={{
                    width: "12px",
                    height: "12px",
                    borderRadius: "50%",
                    backgroundColor: color,
                  }}
                />
                <span style={{ fontSize: "0.875rem", color: "#9ca3af" }}>
                  {type}
                </span>
              </div>
            ))
          : COMMUNITY_COLORS.slice(0, stats.communities).map((color, i) => (
              <div
                key={i}
                style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
              >
                <div
                  style={{
                    width: "12px",
                    height: "12px",
                    borderRadius: "50%",
                    backgroundColor: color,
                  }}
                />
                <span style={{ fontSize: "0.875rem", color: "#9ca3af" }}>
                  Community {i + 1}
                </span>
              </div>
            ))}
      </div>

      {/* Node Details Modal */}
      {selectedNode && (
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
          onClick={() => setSelectedNode(null)}
        >
          <div
            style={{
              backgroundColor: "#1f2937",
              borderRadius: "0.5rem",
              padding: "1.5rem",
              maxWidth: "400px",
              border: "1px solid #374151",
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 style={{ margin: "0 0 1rem 0", color: "#f9fafb" }}>
              {selectedNode.name}
            </h3>
            <div style={{ color: "#9ca3af" }}>
              <p>
                <strong>Type:</strong> {selectedNode.type}
              </p>
              {selectedNode.community !== undefined && (
                <p>
                  <strong>Community:</strong> {selectedNode.community + 1}
                </p>
              )}
              <p>
                <strong>Connections:</strong> {selectedNode.val || 0}
              </p>
            </div>
            <button
              onClick={() => setSelectedNode(null)}
              style={{
                marginTop: "1rem",
                padding: "0.5rem 1rem",
                borderRadius: "0.375rem",
                backgroundColor: "#374151",
                color: "white",
                border: "none",
                cursor: "pointer",
              }}
            >
              Close
            </button>
          </div>
        </div>
      )}
    </main>
  );
}
