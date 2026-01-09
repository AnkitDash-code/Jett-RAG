"use client";

import Sidebar from "@/components/Sidebar";
import { SearchBar } from "@/components/SearchBar";
import { useRouter } from "next/navigation";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();

  const handleSearch = (query: string) => {
    // Navigate to chat with the search query
    router.push(`/chat?q=${encodeURIComponent(query)}`);
  };

  const handleSelectDocument = (docId: string) => {
    // Navigate to document admin page
    router.push(`/admin?doc=${docId}`);
  };

  return (
    <div className="app-layout">
      <Sidebar />
      <div
        className="dashboard-content"
        style={{ display: "flex", flexDirection: "column", flex: 1 }}
      >
        {/* Top Header Bar with Search */}
        <header
          className="dashboard-header"
          style={{
            display: "flex",
            justifyContent: "flex-end",
            alignItems: "center",
            padding: "0.75rem 1.5rem",
            borderBottom: "1px solid #374151",
            backgroundColor: "#111827",
          }}
        >
          <SearchBar
            onSearch={handleSearch}
            onSelectDocument={handleSelectDocument}
            placeholder="Search documents, entities..."
            autoFocus={false}
          />
        </header>
        <div style={{ flex: 1, overflow: "auto" }}>{children}</div>
      </div>
    </div>
  );
}
