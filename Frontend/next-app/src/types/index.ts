export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: Source[];
}

export interface Source {
  id: string;
  name: string;
  page?: number;
  relevance: 'High' | 'Medium' | 'Low';
}

export interface Document {
  id: string;
  name: string;
  size: string;
  uploadDate: Date;
  status: 'indexed' | 'processing' | 'error';
}

export interface User {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'engineer' | 'viewer';
  status: 'active' | 'offline';
}
