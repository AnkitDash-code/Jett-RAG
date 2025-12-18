"use client";

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  ReactNode,
} from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { UserResponse, LoginRequest, SignupRequest } from "@/types/api";

interface AuthContextType {
  user: UserResponse | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (data: LoginRequest) => Promise<void>;
  signup: (data: SignupRequest) => Promise<void>;
  logout: () => Promise<void>;
  refreshAuth: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const TOKEN_KEY = "refresh_token";
const USER_KEY = "user";

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<UserResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();

  // Try to restore session on mount
  useEffect(() => {
    const storedUser = localStorage.getItem(USER_KEY);
    const storedRefreshToken = localStorage.getItem(TOKEN_KEY);

    if (storedUser && storedRefreshToken) {
      try {
        setUser(JSON.parse(storedUser));
        // Try to refresh token silently
        api
          .refreshToken({ refresh_token: storedRefreshToken })
          .then((response) => {
            localStorage.setItem(TOKEN_KEY, response.refresh_token);
            api.setAccessToken(response.access_token);
          })
          .catch(() => {
            // Token expired, clear storage
            localStorage.removeItem(TOKEN_KEY);
            localStorage.removeItem(USER_KEY);
            setUser(null);
          })
          .finally(() => setIsLoading(false));
      } catch {
        setIsLoading(false);
      }
    } else {
      setIsLoading(false);
    }
  }, []);

  const login = useCallback(
    async (data: LoginRequest) => {
      const response = await api.login(data);
      // Transform user: derive 'role' from 'roles' array
      const userWithRole: UserResponse = {
        ...response.user,
        role: response.user.roles?.includes("admin") ? "admin" : "user",
        full_name: response.user.name || response.user.full_name,
        is_active: true,
        status: "active",
        created_at: new Date().toISOString(),
      };
      setUser(userWithRole);
      localStorage.setItem(TOKEN_KEY, response.refresh_token);
      localStorage.setItem(USER_KEY, JSON.stringify(userWithRole));
      router.push("/dashboard");
    },
    [router]
  );

  const signup = useCallback(
    async (data: SignupRequest) => {
      await api.signup(data);
      // Auto-login after signup
      await login({ email: data.email, password: data.password });
    },
    [login]
  );

  const logout = useCallback(async () => {
    const refreshToken = localStorage.getItem(TOKEN_KEY);
    try {
      await api.logout({ refresh_token: refreshToken || undefined });
    } catch {
      // Ignore errors on logout
    }
    setUser(null);
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    api.setAccessToken(null);
    router.push("/sign-in");
  }, [router]);

  const refreshAuth = useCallback(async (): Promise<boolean> => {
    const refreshToken = localStorage.getItem(TOKEN_KEY);
    if (!refreshToken) return false;

    try {
      const response = await api.refreshToken({ refresh_token: refreshToken });
      localStorage.setItem(TOKEN_KEY, response.refresh_token);
      api.setAccessToken(response.access_token);
      return true;
    } catch {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
      setUser(null);
      return false;
    }
  }, []);

  return (
    <AuthContext.Provider
      value={{
        user,
        isAuthenticated: !!user,
        isLoading,
        login,
        signup,
        logout,
        refreshAuth,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
