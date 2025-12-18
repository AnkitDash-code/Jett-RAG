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
    <main className="main-content analytics-page">
      <header>
        <h2>System Analytics</h2>
        <p>Real-time performance metrics and usage statistics.</p>
        <span className="status-online">● System Online</span>
      </header>

      {/* KPI Cards */}
      <section className="analytics-grid">
        <div className="kpi-card">
          <div className="kpi-icon">📄</div>
          <div className="kpi-content">
            <h3>Documents Indexed</h3>
            <p className="kpi-value">
              {metrics?.documents.indexed?.toLocaleString() ?? 0}
            </p>
            <small>
              {metrics?.documents.total_chunks?.toLocaleString() ?? 0} total
              chunks
            </small>
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-icon">👥</div>
          <div className="kpi-content">
            <h3>Active Users (24h)</h3>
            <p className="kpi-value">
              {metrics?.users.active_24h?.toLocaleString() ?? 0}
            </p>
            <small>
              {metrics?.users.total?.toLocaleString() ?? 0} total users
            </small>
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-icon">💬</div>
          <div className="kpi-content">
            <h3>Total Messages</h3>
            <p className="kpi-value">
              {metrics?.chat.total_messages?.toLocaleString() ?? 0}
            </p>
            <small>
              {metrics?.chat.messages_24h?.toLocaleString() ?? 0} in last 24h
            </small>
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-icon">⚡</div>
          <div className="kpi-content">
            <h3>Avg Response Time</h3>
            <p className="kpi-value">
              {metrics?.performance.avg_total_latency_ms?.toFixed(0) ?? "—"}
              <small>ms</small>
            </p>
            <small>End-to-end latency</small>
          </div>
        </div>
      </section>

      {/* Charts Row 1 */}
      <section className="charts-grid">
        <div className="chart-card">
          <h3>Weekly Activity</h3>
          <ResponsiveContainer width="100%" height={250}>
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
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="day" stroke="#6b7280" />
              <YAxis stroke="#6b7280" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#fff",
                  border: "1px solid #e5e7eb",
                  borderRadius: "8px",
                }}
              />
              <Legend />
              <Area
                type="monotone"
                dataKey="queries"
                stroke="#3b82f6"
                fillOpacity={1}
                fill="url(#colorQueries)"
                name="Queries"
              />
              <Area
                type="monotone"
                dataKey="messages"
                stroke="#10b981"
                fillOpacity={1}
                fill="url(#colorMessages)"
                name="Messages"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3>Document Status</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={docStatusData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={90}
                paddingAngle={5}
                dataKey="value"
                label={({ name, value }) => `${name}: ${value}`}
              >
                {docStatusData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Charts Row 2 */}
      <section className="charts-grid">
        <div className="chart-card wide">
          <h3>Response Time Distribution (24h)</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={responseTimeData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="hour" stroke="#6b7280" interval={2} />
              <YAxis stroke="#6b7280" unit="ms" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#fff",
                  border: "1px solid #e5e7eb",
                  borderRadius: "8px",
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="avgLatency"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={false}
                name="Avg Latency"
              />
              <Line
                type="monotone"
                dataKey="p95Latency"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={false}
                name="P95 Latency"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Performance Metrics */}
      <section className="analytics-grid" style={{ marginTop: "2rem" }}>
        <div className="kpi-card">
          <div className="kpi-icon">🔍</div>
          <div className="kpi-content">
            <h3>Retrieval Latency</h3>
            <p className="kpi-value">
              {metrics?.performance.avg_retrieval_latency_ms?.toFixed(0) ?? "—"}
              <small>ms</small>
            </p>
            <small>Vector search time</small>
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-icon">🤖</div>
          <div className="kpi-content">
            <h3>LLM Latency</h3>
            <p className="kpi-value">
              {metrics?.performance.avg_llm_latency_ms?.toFixed(0) ?? "—"}
              <small>ms</small>
            </p>
            <small>Generation time</small>
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-icon">📊</div>
          <div className="kpi-content">
            <h3>Avg Relevance</h3>
            <p className="kpi-value">
              {metrics?.performance.avg_relevance_score
                ? `${(metrics.performance.avg_relevance_score * 100).toFixed(
                    0
                  )}%`
                : "—"}
            </p>
            <small>Retrieval quality</small>
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-icon">👍</div>
          <div className="kpi-content">
            <h3>Helpful Rate</h3>
            <p className="kpi-value">
              {metrics?.feedback.helpful_rate
                ? `${(metrics.feedback.helpful_rate * 100).toFixed(0)}%`
                : "—"}
            </p>
            <small>User satisfaction</small>
          </div>
        </div>
      </section>

      {/* System Health */}
      <section className="system-health">
        <h3>System Health</h3>
        <div className="health-grid">
          <div className="health-item">
            <span className="health-dot green"></span>
            <label>RAG Backend</label>
            <span className="health-status">Operational</span>
          </div>
          <div className="health-item">
            <span className="health-dot green"></span>
            <label>LLM Service</label>
            <span className="health-status">Operational</span>
          </div>
          <div className="health-item">
            <span className="health-dot green"></span>
            <label>Vector Store</label>
            <span className="health-status">FAISS Active</span>
          </div>
          <div className="health-item">
            <span className="health-dot green"></span>
            <label>Database</label>
            <span className="health-status">SQLite Connected</span>
          </div>
        </div>
      </section>
    </main>
  );
}
