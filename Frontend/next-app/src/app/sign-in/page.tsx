"use client";

import { useState } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";

export default function SignIn() {
  const { login, isLoading } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!email || !password) {
      toast.error("Please fill in all fields");
      return;
    }

    setIsSubmitting(true);
    try {
      await login({ email, password });
      toast.success("Welcome back!");
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : "Login failed. Please try again."
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isLoading) {
    return (
      <div className="auth-body">
        <div className="auth-box">
          <h1>GraphRAG</h1>
          <p>Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="auth-body">
      <div className="auth-box">
        <h1>GraphRAG</h1>
        <form className="auth-form" onSubmit={handleSubmit}>
          <h2>Sign In</h2>
          <input
            type="email"
            placeholder="your@email.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            disabled={isSubmitting}
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={isSubmitting}
            required
          />

          <button
            type="submit"
            className="btn btn-dark"
            disabled={isSubmitting}
          >
            {isSubmitting ? "Signing in..." : "Continue"}
          </button>

          <div className="divider">
            <span className="or-text">or</span>
            <div className="line"></div>
          </div>

          <Link href="/create-account" className="link-btn">
            Create An Account
          </Link>

          <p className="policy-text">
            By clicking continue, you agree to our Terms of Service and Privacy
            Policy
          </p>

          <Link href="/" className="back-link">
            ← Go Back
          </Link>
        </form>
      </div>
    </div>
  );
}
