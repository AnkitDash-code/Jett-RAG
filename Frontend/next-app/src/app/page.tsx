"use client";
import Link from "next/link";
import Image from "next/image";
import { motion } from "framer-motion";
import RippleGrid from "@/components/RippleGrid";
import FloatingNavbar from "@/components/FloatingNavbar";
import MagicBento from "@/components/MagicBento";
import PipelineSteps from "@/components/PipelineSteps";



export default function Home() {
  return (
    <main style={{ minHeight: '100vh', backgroundColor: '#000000', width: '100%', color: 'white' }}>
      <FloatingNavbar />

      {/* --- Top Section: Hero + Ripple Grid --- */}
      <section style={{ position: 'relative', width: '100%', minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>

        {/* Ripple Grid Animated Background (Absolute to Top Section) */}
        <div style={{ position: 'absolute', inset: 0, zIndex: 0 }}>
          <RippleGrid
            enableRainbow={false}
            gridColor="#3b82f6"
            rippleIntensity={0.04}
            gridSize={10}
            gridThickness={15}
            mouseInteraction={true}
            mouseInteractionRadius={1.2}
            opacity={0.4}
          />
        </div>

        {/* Hero Content */}
        <div className="landing-container" style={{ position: 'relative', zIndex: 1, margin: '0 auto' }}>
          <motion.div
            className="hero-section"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >

            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              style={{ color: '#ffffff' }}
            >
              Jett-RAG
            </motion.h1>

            <motion.p
              className="subtitle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6 }}
              style={{ color: '#d1d5db' }}
            >
              Secure, Offline, Explainable Document Intelligence
            </motion.p>
            <motion.div className="status-pills">
              {[
                { label: "Offline First", color: "#f59e0b" },
                { label: "Enterprise Secure", color: "#10b981" },
                { label: "Graph Intelligence", color: "#3b82f6" }
              ].map((p, i) => (
                <motion.span
                  key={i}
                  className="pill"
                  whileHover={{ scale: 1.05, backgroundColor: "rgba(255, 255, 255, 0.1)", borderColor: p.color }}
                  whileTap={{ scale: 0.95 }}
                  style={{ backgroundColor: 'rgba(255,255,255,0.05)', color: '#e5e7eb', borderColor: 'rgba(255,255,255,0.1)' }}
                >
                  <span className="dot" style={{ backgroundColor: p.color }}></span> {p.label}
                </motion.span>
              ))}
            </motion.div>
            <motion.p
              className="description"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1 }}
              style={{ color: '#9ca3af' }}
            >
              Transform your static documents into an interactive knowledge graph.
              Ask complex questions and get precise, sourced answers without your
              data ever leaving your infrastructure.
            </motion.p>

            <motion.div
              className="cta-group"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.2 }}
            >
              <Link href="/sign-in" className="btn btn-dark btn-lg" style={{ backgroundColor: '#ffffff', color: '#000000' }}>
                Get Started
              </Link>
              <Link href="/about" className="btn btn-light btn-lg" style={{ backgroundColor: '#0D0F16', color: '#ffffff', borderColor: 'rgba(255,255,255,0.08)' }}>
                Learn More
              </Link>
            </motion.div>
          </motion.div>

          {/* Features Grid */}
          <motion.div
            className="features-grid"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1.4 }}
          >
            <div className="feature-card-modern">
              <div className="feature-card-image-container">
                <div className="feature-card-glow" />
                <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: '180px' }}>
                  <Image
                    src="/documents-icon.png"
                    alt="Document Processing"
                    fill
                    style={{ objectFit: 'contain', zIndex: 1 }}
                    className="drag-none"
                  />
                </div>
              </div>
              <div>
                <h3>Document Processing</h3>
                <p>
                  Upload PDFs and let our AI structure unstructured data
                  automatically.
                </p>
              </div>
            </div>
            <div className="feature-card-modern">
              <div className="feature-card-image-container">
                <div className="feature-card-glow" />
                <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: '180px' }}>
                  <Image
                    src="/privacy-icon.png"
                    alt="Privacy by Design"
                    fill
                    style={{ objectFit: 'contain', zIndex: 1 }}
                    className="drag-none"
                  />
                </div>
              </div>
              <div>
                <h3>Privacy by Design</h3>
                <p>
                  Zero data egress. Your knowledge base runs entirely on your local
                  hardware.
                </p>
              </div>
            </div>
            <div className="feature-card-modern">
              <div className="feature-card-image-container">
                <div className="feature-card-glow" />
                <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: '180px' }}>
                  <Image
                    src="/brain-icon.png"
                    alt="Graph Reasoning"
                    fill
                    style={{ objectFit: 'contain', zIndex: 1, transform: 'scale(1.5)' }}
                    className="drag-none"
                  />
                </div>
              </div>
              <div>
                <h3>Graph Reasoning</h3>
                <p>
                  Go beyond keyword search with semantic understanding and
                  relationship mapping.
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* --- Bottom Section: Architecture / How it Works --- */}
      {/* --- Bottom Section: Magic Bento --- */}
      <section
        style={{
          position: 'relative',
          padding: '6rem 2rem',
          background: 'linear-gradient(180deg, #000000 0%, #020617 30%, #172554 100%)',
          zIndex: 1,
          width: '100%'
        }}
      >
        <div style={{ maxWidth: '1200px', margin: '0 auto', textAlign: 'center' }}>
          <h2 style={{
            fontSize: '2.5rem',
            fontWeight: '700',
            marginBottom: '1rem',
            background: 'linear-gradient(to right, #ffffff, #94a3b8)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>
            Powering Next-Gen RAG
          </h2>
          <p style={{
            color: '#9ca3af',
            fontSize: '1.2rem',
            marginBottom: '4rem',
            maxWidth: '600px',
            marginLeft: 'auto',
            marginRight: 'auto'
          }}>
            A complete ecosystem for building intelligent, graph-aware AI applications.
          </p>

          <MagicBento
            textAutoHide={true}
            enableStars={true}
            enableSpotlight={true}
            enableBorderGlow={true}
            enableTilt={true}
            enableMagnetism={true}
            clickEffect={true}
            spotlightRadius={300}
            particleCount={12}
            glowColor="59, 130, 246"
          />
        </div>
      </section>

      {/* --- New Pipeline Steps Section (Full Width) --- */}
      <div style={{ marginTop: '4rem' }}>
        <PipelineSteps />
      </div>

      {/* --- Footer (Full Width Section) --- */}
      <footer className="landing-footer" style={{ borderTop: '1px solid rgba(255,255,255,0.1)', marginTop: '0', paddingTop: '4rem', paddingBottom: '4rem', backgroundColor: '#000000' }}>
        <div className="footer-content" style={{ color: '#9ca3af', maxWidth: '1200px', margin: '0 auto', padding: '0 2rem' }}>
          <div className="footer-col">
            <h4 style={{ color: '#ffffff' }}>GraphRAG</h4>
            <p>Secure, offline document intelligence for the enterprise.</p>
          </div>
          <div className="footer-col">
            <h4 style={{ color: '#ffffff' }}>Product</h4>
            <Link href="#" className="hover:text-white">Features</Link>
            <Link href="#" className="hover:text-white">Security</Link>
            <Link href="#" className="hover:text-white">Enterprise</Link>
          </div>
          <div className="footer-col">
            <h4 style={{ color: '#ffffff' }}>Resources</h4>
            <Link href="#" className="hover:text-white">Documentation</Link>
            <Link href="#" className="hover:text-white">API Reference</Link>
            <Link href="#" className="hover:text-white">Blog</Link>
          </div>
          <div className="footer-col">
            <h4 style={{ color: '#ffffff' }}>Company</h4>
            <Link href="#" className="hover:text-white">About</Link>
            <Link href="#" className="hover:text-white">Contact</Link>
            <Link href="#" className="hover:text-white">Privacy</Link>
          </div>
        </div>
        <div className="footer-bottom" style={{ color: '#6b7280', borderTop: '1px solid rgba(255,255,255,0.05)', marginTop: '2rem', maxWidth: '1200px', marginLeft: 'auto', marginRight: 'auto', padding: '2rem 2rem 0' }}>
          <p>
            &copy; {new Date().getFullYear()} GraphRAG. All rights reserved.
          </p>
        </div>
      </footer>
    </main>
  );
}

