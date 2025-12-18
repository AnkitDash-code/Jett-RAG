import Link from "next/link";

export default function Home() {
  return (
    <div className="landing-body">
      <div className="landing-container">
        <div className="hero-section">
          <div className="logo-badge">G</div>
          <h1>GraphRAG Knowledge Portal</h1>
          <p className="subtitle">
            Secure, Offline, Explainable Document Intelligence
          </p>

          <div className="status-pills">
            <span className="pill offline">
              <span className="dot"></span> Offline First
            </span>
            <span className="pill secure">
              <span className="dot"></span> Enterprise Secure
            </span>
            <span className="pill ai">
              <span className="dot"></span> Graph Intelligence
            </span>
          </div>

          <p className="description">
            Transform your static documents into an interactive knowledge graph.
            Ask complex questions and get precise, sourced answers without your
            data ever leaving your infrastructure.
          </p>

          <div className="cta-group">
            <Link href="/sign-in" className="btn btn-dark btn-lg">
              Get Started
            </Link>
            <Link href="/about" className="btn btn-light btn-lg">
              Learn More
            </Link>
          </div>
        </div>

        <div className="features-grid">
          <div className="feature-card">
            <div className="icon-box">📄</div>
            <h3>Document Processing</h3>
            <p>
              Upload PDFs and let our AI structure unstructured data
              automatically.
            </p>
          </div>
          <div className="feature-card">
            <div className="icon-box">🛡️</div>
            <h3>Privacy by Design</h3>
            <p>
              Zero data egress. Your knowledge base runs entirely on your local
              hardware.
            </p>
          </div>
          <div className="feature-card">
            <div className="icon-box">🧠</div>
            <h3>Graph Reasoning</h3>
            <p>
              Go beyond keyword search with semantic understanding and
              relationship mapping.
            </p>
          </div>
        </div>
      </div>

      <footer className="landing-footer">
        <div className="footer-content">
          <div className="footer-col">
            <h4>GraphRAG</h4>
            <p>Secure, offline document intelligence for the enterprise.</p>
          </div>
          <div className="footer-col">
            <h4>Product</h4>
            <Link href="#">Features</Link>
            <Link href="#">Security</Link>
            <Link href="#">Enterprise</Link>
          </div>
          <div className="footer-col">
            <h4>Resources</h4>
            <Link href="#">Documentation</Link>
            <Link href="#">API Reference</Link>
            <Link href="#">Blog</Link>
          </div>
          <div className="footer-col">
            <h4>Company</h4>
            <Link href="#">About</Link>
            <Link href="#">Contact</Link>
            <Link href="#">Privacy</Link>
          </div>
        </div>
        <div className="footer-bottom">
          <p>
            &copy; {new Date().getFullYear()} GraphRAG. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}
