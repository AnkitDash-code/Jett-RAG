"use client";
import React, { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { motion, useScroll, useMotionValueEvent } from "framer-motion";

export default function FloatingNavbar() {
  const { scrollY } = useScroll();
  const [isScrolled, setIsScrolled] = useState(false);

  useMotionValueEvent(scrollY, "change", (latest) => {
    if (latest > 50) {
      setIsScrolled(true);
    } else {
      setIsScrolled(false);
    }
  });

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        zIndex: 100,
        display: "flex",
        justifyContent: "center",
        paddingTop: isScrolled ? "1rem" : "0",
        transition: "padding-top 0.3s ease",
        pointerEvents: "none",
      }}
    >
      <motion.nav
        initial={{
          width: "100%",
          borderRadius: "0px",
          backgroundColor: "rgba(0,0,0,0)",
          borderBottom: "1px solid rgba(255,255,255,0.05)",
          y: 0,
        }}
        animate={{
          width: isScrolled ? "90%" : "100%",
          borderRadius: isScrolled ? "16px" : "0px",
          backgroundColor: isScrolled
            ? "rgba(20, 20, 30, 0.6)"
            : "rgba(0, 0, 0, 0)",
          backdropFilter: isScrolled ? "blur(12px)" : "blur(0px)",
          borderBottom: isScrolled
            ? "1px solid rgba(59, 130, 246, 0.2)"
            : "1px solid rgba(255,255,255,0.05)",
          borderColor: isScrolled
            ? "rgba(59, 130, 246, 0.3)"
            : "rgba(255,255,255,0.05)",
          y: isScrolled ? 0 : 0,
        }}
        transition={{ duration: 0.3, ease: "easeInOut" }}
        style={{
          height: "70px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 2rem",
          width: isScrolled ? "90%" : "100%",
          gap: "2rem",
          pointerEvents: "auto",
          boxShadow: isScrolled ? "0 10px 30px -10px rgba(0,0,0,0.5)" : "none",
          whiteSpace: "nowrap",
        }}
      >
        {/* Logo Area */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <div
            style={{
              width: "40px",
              height: "40px",
              position: "relative",
              flexShrink: 0,
            }}
          >
            <Image
              src="/jett-rag-logo.png"
              alt="Jett-RAG Logo"
              fill
              style={{ objectFit: "contain" }}
            />
          </div>
          <span
            style={{ color: "white", fontWeight: "600", fontSize: "1.1rem" }}
          >
            Jett-RAG
          </span>
        </div>

        {/* Links */}
        <div
          className="nav-links"
          style={{
            display: "flex",
            gap: "2rem",
            alignItems: "center",
            flexShrink: 0,
          }}
        >
          {["Features", "Security", "Enterprise", "Docs"].map((item) => (
            <Link
              key={item}
              href="#"
              style={{
                color: "#d1d5db",
                textDecoration: "none",
                fontSize: "0.95rem",
                transition: "color 0.2s",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "white")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "#d1d5db")}
            >
              {item}
            </Link>
          ))}
        </div>

        {/* CTA */}
        <Link
          href="/sign-in"
          style={{
            backgroundColor: "#ffffff",
            color: "#000000",
            padding: "0.5rem 1.25rem",
            borderRadius: "8px",
            textDecoration: "none",
            fontWeight: "500",
            fontSize: "0.9rem",
            transition: "all 0.2s",
            border: "1px solid rgba(255,255,255,0.1)",
            flexShrink: 0,
          }}
        >
          Sign In
        </Link>
      </motion.nav>
    </div>
  );
}
