"use client";

import { useState } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { useAuth } from "@/contexts/AuthContext";
import RippleGrid from "@/components/RippleGrid";

export default function CreateAccount() {
  const { signup, isLoading } = useAuth();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Background component for consistency
  const Background = () => (
    <div style={{ position: "absolute", inset: 0, zIndex: 0 }}>
      <RippleGrid
        enableRainbow={false}
        gridColor="#3b82f6"
        rippleIntensity={0.1}
        gridSize={10}
        gridThickness={15}
        mouseInteraction={true}
        mouseInteractionRadius={1.2}
        opacity={0.7}
      />
    </div>
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!email || !password) {
      toast.error("Please fill in all required fields");
      return;
    }

    if (password.length < 8) {
      toast.error("Password must be at least 8 characters");
      return;
    }

    setIsSubmitting(true);
    try {
      await signup({ email, password, name: name || undefined });
      toast.success("Account created successfully!");
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : "Signup failed. Please try again."
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isLoading) {
    return (
      <div className="auth-body">
        <Background />
        <div className="auth-box">
          <h1>GraphRAG</h1>
          <p style={{ color: "#9ca3af" }}>Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="auth-body">
      <Background />
      <div className="auth-box">
        <h1>GraphRAG</h1>
        <form className="auth-form" onSubmit={handleSubmit}>
          <h2>Create an account</h2>
          <input
            type="text"
            placeholder="Name (optional)"
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={isSubmitting}
          />
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
            placeholder="Password (min 8 characters)"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={isSubmitting}
            required
            minLength={8}
          />

          <button
            type="submit"
            className="btn btn-dark"
            disabled={isSubmitting}
          >
            {isSubmitting ? "Creating account..." : "Continue"}
          </button>

          <p className="policy-text">
            By clicking continue, you agree to our Terms of Service and Privacy
            Policy
          </p>

          <Link href="/sign-in" className="back-link">
            ← Go Back
          </Link>
        </form>
      </div>
    </div>
  );
}
