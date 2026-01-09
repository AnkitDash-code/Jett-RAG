import type {
  LoginRequest,
  LoginResponse,
  SignupRequest,
  SignupResponse,
  TokenRefreshRequest,
  TokenRefreshResponse,
  LogoutRequest,
  UserResponse,
  UserUpdateRequest,
  UserListResponse,
  DocumentUploadResponse,
  DocumentResponse,
  DocumentListResponse,
  ChatRequest,
  ChatResponse,
  ConversationResponse,
  ConversationListResponse,
  FeedbackRequest,
  SystemMetrics,
} from "@/types/api";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8081/v1";

class ApiClient {
  private accessToken: string | null = null;

  setAccessToken(token: string | null) {
    this.accessToken = token;
  }

  getAccessToken(): string | null {
    return this.accessToken;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const headers: HeadersInit = {
      "Content-Type": "application/json",
      ...options.headers,
    };

    if (this.accessToken) {
      (headers as Record<string, string>)["Authorization"] =
        `Bearer ${this.accessToken}`;
    }

    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers,
      // Don't use credentials - backend uses allow_credentials=False with wildcard origins
      credentials: "omit",
    });

    if (response.status === 401) {
      throw new Error("Unauthorized");
    }

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `API error: ${response.status}`);
    }

    // Handle empty responses
    const text = await response.text();
    if (!text) return {} as T;
    return JSON.parse(text);
  }

  // ============== AUTH ==============
  async login(data: LoginRequest): Promise<LoginResponse> {
    const response = await this.request<LoginResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify(data),
    });
    this.accessToken = response.access_token;
    return response;
  }

  async signup(data: SignupRequest): Promise<SignupResponse> {
    return this.request<SignupResponse>("/auth/signup", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async refreshToken(data: TokenRefreshRequest): Promise<TokenRefreshResponse> {
    const response = await this.request<TokenRefreshResponse>("/auth/refresh", {
      method: "POST",
      body: JSON.stringify(data),
    });
    this.accessToken = response.access_token;
    return response;
  }

  async logout(data?: LogoutRequest): Promise<void> {
    await this.request("/auth/logout", {
      method: "POST",
      body: JSON.stringify(data || {}),
    });
    this.accessToken = null;
  }

  // ============== USER ==============
  async getCurrentUser(): Promise<UserResponse> {
    return this.request<UserResponse>("/users/me");
  }

  async updateCurrentUser(data: UserUpdateRequest): Promise<UserResponse> {
    return this.request<UserResponse>("/users/me", {
      method: "PATCH",
      body: JSON.stringify(data),
    });
  }

  async listUsers(params?: {
    page?: number;
    page_size?: number;
  }): Promise<UserListResponse> {
    const query = new URLSearchParams(
      params as Record<string, string>
    ).toString();
    return this.request<UserListResponse>(`/admin/users?${query}`);
  }

  // ============== DOCUMENTS ==============
  async uploadDocument(
    file: File,
    metadata?: { title?: string; tags?: string }
  ): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append("file", file);
    if (metadata?.title) formData.append("title", metadata.title);
    if (metadata?.tags) formData.append("tags", metadata.tags);

    const headers: HeadersInit = {};
    if (this.accessToken) {
      headers["Authorization"] = `Bearer ${this.accessToken}`;
    }

    const response = await fetch(`${API_BASE_URL}/documents/upload`, {
      method: "POST",
      headers,
      body: formData,
      credentials: "omit",
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "Upload failed");
    }

    return response.json();
  }

  async listDocuments(params?: {
    page?: number;
    status?: string;
  }): Promise<DocumentListResponse> {
    const query = new URLSearchParams(
      params as Record<string, string>
    ).toString();
    return this.request<DocumentListResponse>(`/documents?${query}`);
  }

  async getDocument(id: string): Promise<DocumentResponse> {
    return this.request<DocumentResponse>(`/documents/${id}`);
  }

  async deleteDocument(id: string): Promise<void> {
    await this.request(`/documents/${id}`, { method: "DELETE" });
  }

  // ============== CHAT ==============
  async sendMessage(data: ChatRequest): Promise<ChatResponse> {
    return this.request<ChatResponse>("/chat", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async getChatHistory(params?: {
    page?: number;
  }): Promise<ConversationListResponse> {
    const query = new URLSearchParams(
      params as Record<string, string>
    ).toString();
    return this.request<ConversationListResponse>(`/chat/history?${query}`);
  }

  async getConversation(id: string): Promise<ConversationResponse> {
    return this.request<ConversationResponse>(`/chat/conversations/${id}`);
  }

  async submitFeedback(data: FeedbackRequest): Promise<void> {
    await this.request("/chat/feedback", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  // ============== ADMIN ==============
  async getMetrics(): Promise<SystemMetrics> {
    return this.request<SystemMetrics>("/admin/metrics");
  }

  async getUsers(): Promise<UserListResponse> {
    return this.request<UserListResponse>("/admin/users");
  }

  async updateUser(
    userId: string,
    data: UserUpdateRequest
  ): Promise<UserResponse> {
    return this.request<UserResponse>(`/users/${userId}`, {
      method: "PUT",
      body: JSON.stringify(data),
    });
  }

  async getUserMetrics(userId: string): Promise<Record<string, unknown>> {
    return this.request(`/admin/metrics/${userId}`);
  }

  async deactivateUser(userId: string): Promise<void> {
    await this.request(`/admin/users/${userId}/deactivate`, { method: "POST" });
  }

  // ============== STREAMING ==============
  getStreamUrl(message: string, conversationId?: string): string {
    const params = new URLSearchParams({ message });
    if (conversationId) params.append("conversation_id", conversationId);
    return `${API_BASE_URL}/chat/stream?${params}`;
  }
}

export const api = new ApiClient();
export { API_BASE_URL };
