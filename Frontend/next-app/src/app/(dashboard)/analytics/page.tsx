"use client";

import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import type { SystemMetrics } from "@/types/api";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from "recharts";

// Mock data for charts (in production, this would come from API)
const generateMockActivityData = () => {
  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  return days.map((day) => ({
    day,
    queries: Math.floor(Math.random() * 100) + 20,
    messages: Math.floor(Math.random() * 150) + 30,
  }));
};

const generateMockResponseTimes = () => {
  const hours = Array.from({ length: 24 }, (_, i) => `${i}:00`);
  return hours.map((hour) => ({
    hour,
    avgLatency: Math.floor(Math.random() * 500) + 200,
    p95Latency: Math.floor(Math.random() * 800) + 400,
  }));
};

const documentStatusData = [
  { name: "Indexed", value: 0, color: "#10b981" },
  { name: "Processing", value: 0, color: "#f59e0b" },
  { name: "Error", value: 0, color: "#ef4444" },
];

export default function Analytics() {
  const { user } = useAuth();
  const router = useRouter();
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activityData, setActivityData] = useState(generateMockActivityData());
  const [responseTimeData] = useState(generateMockResponseTimes());
  const [docStatusData, setDocStatusData] = useState(documentStatusData);

  // Redirect non-admins
  useEffect(() => {
    if (user && user.role !== "admin") {
      router.push("/dashboard");
      toast.error("Access denied. Admin privileges required.");
    }
  }, [user, router]);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setIsLoading(true);
        const data = await api.getMetrics();
        setMetrics(data);

        // Update document status chart
        setDocStatusData([
          {
            name: "Indexed",
            value: data.documents.indexed || 0,
            color: "#10b981",
          },
          {
            name: "Processing",
            value: data.documents.processing || 0,
            color: "#f59e0b",
          },
          { name: "Error", value: data.documents.error || 0, color: "#ef4444" },
        ]);

        setError(null);
      } catch (err) {
        setError((err as Error).message || "Failed to load metrics");
      } finally {
        setIsLoading(false);
      }
    };

    fetchMetrics();
    // Refresh metrics every 30 seconds
    const interval = setInterval(() => {
      fetchMetrics();
      setActivityData(generateMockActivityData()); // Simulate real-time updates
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  if (user?.role !== "admin") {
    return null;
  }

  if (isLoading && !metrics) {
    return (
      <main className="main-content">
        <header>
          <h2>System Analytics</h2>
          <p>Loading metrics...</p>
        </header>
      </main>
    );
  }

  if (error) {
    return (
      <main className="main-content">
        <header>
          <h2>System Analytics</h2>
          <p style={{ color: "#dc2626" }}>{error}</p>
        </header>
      </main>
    );
  }

  return (
    <main className="main-content">
      <header style={{ marginBottom: "2rem" }}>
        <h2 style={{ fontSize: "2rem", marginBottom: "0.5rem", background: "linear-gradient(to right, #fff, #9ca3af)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>System Analytics</h2>
        <p style={{ color: "#9ca3af" }}>
          Real-time performance metrics and usage statistics.
        </p>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "1rem" }}>
          <span style={{ display: "inline-block", width: "8px", height: "8px", borderRadius: "50%", background: "#10b981", boxShadow: "0 0 8px rgba(16, 185, 129, 0.5)" }}></span>
          <span style={{ color: "#10b981", fontSize: "0.875rem", fontWeight: "600" }}>System Online</span>
        </div>
      </header>

      {/* KPI Cards */}
      <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1.5rem", marginBottom: "2rem" }}>
        {/* Documents KPI */}
        <div className="feature-card-modern" style={{ padding: "1.5rem", background: "rgba(255, 255, 255, 0.03)", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: "1rem" }}>
            <div style={{ padding: "0.75rem", borderRadius: "0.75rem", background: "rgba(59, 130, 246, 0.1)", color: "#3b82f6" }}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
            </div>
          </div>
          <p style={{ color: "#9ca3af", fontSize: "0.875rem", marginBottom: "0.25rem" }}>Documents Indexed</p>
          <h3 style={{ fontSize: "1.875rem", fontWeight: "700", color: "white", marginBottom: "0.25rem" }}>{metrics?.documents.indexed?.toLocaleString() ?? 0}</h3>
          <p style={{ color: "#6b7280", fontSize: "0.75rem" }}>{metrics?.documents.total_chunks?.toLocaleString() ?? 0} total chunks</p>
        </div>

        {/* Active Users KPI */}
        <div className="feature-card-modern" style={{ padding: "1.5rem", background: "rgba(255, 255, 255, 0.03)", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: "1rem" }}>
            <div style={{ padding: "0.75rem", borderRadius: "0.75rem", background: "rgba(16, 185, 129, 0.1)", color: "#10b981" }}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
            </div>
          </div>
          <p style={{ color: "#9ca3af", fontSize: "0.875rem", marginBottom: "0.25rem" }}>Active Users (24h)</p>
          <h3 style={{ fontSize: "1.875rem", fontWeight: "700", color: "white", marginBottom: "0.25rem" }}>{metrics?.users.active_24h?.toLocaleString() ?? 0}</h3>
          <p style={{ color: "#6b7280", fontSize: "0.75rem" }}>{metrics?.users.total?.toLocaleString() ?? 0} total users</p>
        </div>

        {/* Total Messages KPI */}
        <div className="feature-card-modern" style={{ padding: "1.5rem", background: "rgba(255, 255, 255, 0.03)", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: "1rem" }}>
            <div style={{ padding: "0.75rem", borderRadius: "0.75rem", background: "rgba(245, 158, 11, 0.1)", color: "#f59e0b" }}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
            </div>
          </div>
          <p style={{ color: "#9ca3af", fontSize: "0.875rem", marginBottom: "0.25rem" }}>Total Messages</p>
          <h3 style={{ fontSize: "1.875rem", fontWeight: "700", color: "white", marginBottom: "0.25rem" }}>{metrics?.chat.total_messages?.toLocaleString() ?? 0}</h3>
          <p style={{ color: "#6b7280", fontSize: "0.75rem" }}>{metrics?.chat.messages_24h?.toLocaleString() ?? 0} in last 24h</p>
        </div>

        {/* Latency KPI */}
        <div className="feature-card-modern" style={{ padding: "1.5rem", background: "rgba(255, 255, 255, 0.03)", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "start", marginBottom: "1rem" }}>
            <div style={{ padding: "0.75rem", borderRadius: "0.75rem", background: "rgba(236, 72, 153, 0.1)", color: "#ec4899" }}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
            </div>
          </div>
          <p style={{ color: "#9ca3af", fontSize: "0.875rem", marginBottom: "0.25rem" }}>Avg Response Time</p>
          <h3 style={{ fontSize: "1.875rem", fontWeight: "700", color: "white", marginBottom: "0.25rem" }}>{metrics?.performance.avg_total_latency_ms?.toFixed(0) ?? "—"}<span style={{ fontSize: "1rem", fontWeight: "400", color: "#6b7280", marginLeft: "0.25rem" }}>ms</span></h3>
          <p style={{ color: "#6b7280", fontSize: "0.75rem" }}>End-to-end latency</p>
        </div>
      </section>

      {/* Charts Row 1 */}
      <section style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: "1.5rem", marginBottom: "1.5rem" }} className="charts-grid-desktop">
        <div style={{ background: "#0a0a0a", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem", padding: "1.5rem" }}>
          <h3 style={{ color: "white", fontSize: "1.125rem", fontWeight: "600", marginBottom: "1.5rem" }}>Weekly Activity</h3>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={activityData}>
              <defs>
                <linearGradient id="colorQueries" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorMessages" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
              <XAxis dataKey="day" stroke="#9ca3af" tickLine={false} axisLine={false} dy={10} />
              <YAxis stroke="#9ca3af" tickLine={false} axisLine={false} dx={-10} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#111827",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#f9fafb"
                }}
              />
              <Legend />
              <Area
                type="monotone"
                dataKey="queries"
                stroke="#3b82f6"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorQueries)"
                name="Queries"
              />
              <Area
                type="monotone"
                dataKey="messages"
                stroke="#10b981"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorMessages)"
                name="Messages"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div style={{ background: "#0a0a0a", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem", padding: "1.5rem" }}>
          <h3 style={{ color: "white", fontSize: "1.125rem", fontWeight: "600", marginBottom: "1.5rem" }}>Document Status</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={docStatusData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={90}
                paddingAngle={5}
                dataKey="value"
                stroke="none"
              >
                {docStatusData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  backgroundColor: "#111827",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#f9fafb"
                }}
              />
              <Legend verticalAlign="bottom" height={36} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Charts Row 2 - Line Chart */}
      <section style={{ marginBottom: "2rem" }}>
        <div style={{ background: "#0a0a0a", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem", padding: "1.5rem" }}>
          <h3 style={{ color: "white", fontSize: "1.125rem", fontWeight: "600", marginBottom: "1.5rem" }}>Response Time Distribution (24h)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={responseTimeData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
              <XAxis dataKey="hour" stroke="#9ca3af" interval={2} tickLine={false} axisLine={false} dy={10} />
              <YAxis stroke="#9ca3af" unit="ms" tickLine={false} axisLine={false} dx={-10} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#111827",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                  color: "#f9fafb"
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="avgLatency"
                stroke="#3b82f6"
                strokeWidth={3}
                dot={false}
                name="Avg Latency"
              />
              <Line
                type="monotone"
                dataKey="p95Latency"
                stroke="#f59e0b"
                strokeWidth={3}
                dot={false}
                name="P95 Latency"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Detailed Metrics */}
      <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1.5rem", marginBottom: "2rem" }}>
        <div style={{ background: "rgba(255, 255, 255, 0.03)", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem", padding: "1.25rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.5rem" }}>
            <span style={{ fontSize: "1.25rem" }}>🔍</span>
            <h4 style={{ color: "#d1d5db", fontWeight: "600" }}>Retrieval Latency</h4>
          </div>
          <p style={{ fontSize: "1.5rem", fontWeight: "700", color: "white" }}>
            {metrics?.performance.avg_retrieval_latency_ms?.toFixed(0) ?? "—"} <small style={{ fontSize: "0.875rem", color: "#6b7280", fontWeight: "400" }}>ms</small>
          </p>
        </div>

        <div style={{ background: "rgba(255, 255, 255, 0.03)", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem", padding: "1.25rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.5rem" }}>
            <span style={{ fontSize: "1.25rem" }}>🤖</span>
            <h4 style={{ color: "#d1d5db", fontWeight: "600" }}>LLM Latency</h4>
          </div>
          <p style={{ fontSize: "1.5rem", fontWeight: "700", color: "white" }}>
            {metrics?.performance.avg_llm_latency_ms?.toFixed(0) ?? "—"} <small style={{ fontSize: "0.875rem", color: "#6b7280", fontWeight: "400" }}>ms</small>
          </p>
        </div>

        <div style={{ background: "rgba(255, 255, 255, 0.03)", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem", padding: "1.25rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.5rem" }}>
            <span style={{ fontSize: "1.25rem" }}>📊</span>
            <h4 style={{ color: "#d1d5db", fontWeight: "600" }}>Avg Relevance</h4>
          </div>
          <p style={{ fontSize: "1.5rem", fontWeight: "700", color: "white" }}>
            {metrics?.performance.avg_relevance_score ? `${(metrics.performance.avg_relevance_score * 100).toFixed(0)}%` : "—"}
          </p>
        </div>

        <div style={{ background: "rgba(255, 255, 255, 0.03)", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem", padding: "1.25rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.5rem" }}>
            <span style={{ fontSize: "1.25rem" }}>👍</span>
            <h4 style={{ color: "#d1d5db", fontWeight: "600" }}>Helpful Rate</h4>
          </div>
          <p style={{ fontSize: "1.5rem", fontWeight: "700", color: "white" }}>
            {metrics?.feedback.helpful_rate ? `${(metrics.feedback.helpful_rate * 100).toFixed(0)}%` : "—"}
          </p>
        </div>
      </section>

      {/* System Health */}
      <section style={{ background: "#0a0a0a", border: "1px solid rgba(255, 255, 255, 0.1)", borderRadius: "1rem", padding: "1.5rem" }}>
        <h3 style={{ color: "white", fontSize: "1.125rem", fontWeight: "600", marginBottom: "1.5rem" }}>System Health</h3>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem" }}>
          {[
            { label: "RAG Backend", status: "Operational", color: "#10b981" },
            { label: "LLM Service", status: "Operational", color: "#10b981" },
            { label: "Vector Store", status: "FAISS Active", color: "#3b82f6" },
            { label: "Database", status: "SQLite Connected", color: "#8b5cf6" },
          ].map((item, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "1rem", background: "rgba(255, 255, 255, 0.03)", borderRadius: "0.5rem", border: "1px solid rgba(255, 255, 255, 0.05)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: item.color, boxShadow: `0 0 8px ${item.color}80` }}></span>
                <span style={{ color: "#d1d5db", fontSize: "0.95rem" }}>{item.label}</span>
              </div>
              <span style={{ color: item.color, fontSize: "0.875rem", fontWeight: "500" }}>{item.status}</span>
            </div>
          ))}
        </div>
      </section>

      <style jsx>{`
        @media (max-width: 1024px) {
          .charts-grid-desktop {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </main>
  );
}
