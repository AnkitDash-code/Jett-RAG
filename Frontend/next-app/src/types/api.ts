// ============== AUTH ==============
export interface SignupRequest {
  email: string;
  password: string;
  name?: string;
}

export interface SignupResponse {
  id: string;
  email: string;
  name?: string;
  message: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: UserResponse;
}

export interface TokenRefreshRequest {
  refresh_token: string;
}

export interface TokenRefreshResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface LogoutRequest {
  refresh_token?: string;
  logout_all_devices?: boolean;
}

// ============== USER ==============
export interface UserResponse {
  id: string;
  email: string;
  full_name?: string;
  name?: string;
  avatar_url?: string;
  status: "active" | "inactive" | "suspended";
  is_active: boolean;
  role: "admin" | "user";
  roles: string[];
  tenant_id?: string;
  department?: string;
  created_at: string;
  last_login_at?: string;
}

export interface UserUpdateRequest {
  name?: string;
  avatar_url?: string;
  department?: string;
  preferences?: Record<string, unknown>;
  role?: "admin" | "user";
  is_active?: boolean;
}

export interface UserListResponse {
  users: UserResponse[];
  total: number;
  page: number;
  page_size: number;
  pages: number;
}

// ============== DOCUMENT ==============
export interface DocumentUploadResponse {
  id: string;
  filename: string;
  status: string;
  message: string;
}

export interface DocumentResponse {
  id: string;
  filename: string;
  original_filename: string;
  file_size: number;
  mime_type: string;
  title?: string;
  description?: string;
  tags: string[];
  status: "pending" | "processing" | "indexed" | "error";
  access_level: string;
  page_count?: number;
  chunk_count: number;
  created_at: string;
  updated_at: string;
  indexed_at?: string;
}

export interface DocumentListResponse {
  documents: DocumentResponse[];
  total: number;
  page: number;
  page_size: number;
  pages: number;
}

// ============== CHAT ==============
export interface SourceCitation {
  id: string;
  name: string;
  page?: number;
  section?: string;
  relevance: "High" | "Medium" | "Low";
  snippet?: string;
}

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  document_ids?: string[];
  date_range?: { start: string; end: string };
  tags?: string[];
  temperature?: number;
  max_tokens?: number;
}

export interface ChatResponse {
  conversation_id: string;
  message_id: string;
  content: string;
  sources: SourceCitation[];
  model: string;
  tokens_used: number;
  latency_ms: number;
}

export interface ChatStreamEvent {
  event: "token" | "sources" | "done" | "error";
  data: string;
  conversation_id?: string;
  message_id?: string;
}

export interface MessageResponse {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  sources?: SourceCitation[];
  created_at: string;
}

export interface ConversationResponse {
  id: string;
  title?: string;
  messages: MessageResponse[];
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ConversationSummary {
  id: string;
  title?: string;
  message_count: number;
  created_at: string;
  last_message_at?: string;
}

export interface ConversationListResponse {
  conversations: ConversationSummary[];
  total: number;
  page: number;
  page_size: number;
}

export interface FeedbackRequest {
  message_id: string;
  rating: number;
  is_helpful?: boolean;
  is_accurate?: boolean;
  comment?: string;
  issues?: string[];
}

// ============== ADMIN ==============
export interface SystemMetrics {
  users: { total: number; active_24h: number };
  documents: {
    total: number;
    processing: number;
    indexed: number;
    error: number;
    total_chunks: number;
  };
  chat: {
    total_conversations: number;
    total_messages: number;
    messages_24h: number;
  };
  performance: {
    avg_retrieval_latency_ms: number;
    avg_llm_latency_ms: number;
    avg_total_latency_ms: number;
    avg_relevance_score: number;
  };
  feedback: {
    total: number;
    avg_rating: number;
    helpful_rate: number;
  };
  generated_at: string;
}

export interface QueryLog {
  id: string;
  user_id: string;
  query: string;
  response_time_ms: number;
  created_at: string;
}

// ============== API ERROR ==============
export interface ApiError {
  detail: string;
  status_code?: number;
}
