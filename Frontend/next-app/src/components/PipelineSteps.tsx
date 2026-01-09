"use client";
import React from "react";
import Image from "next/image";
import { motion } from "framer-motion";
import { ArrowRight, FileText, Share2, Database, Search } from "lucide-react";
import "./PipelineSteps.css";

const steps = [
  {
    id: 1,
    step: "01 - INGEST",
    title: "Raw Data Ingestion",
    description:
      "Upload PDFs, Docs, or connect to APIs. The system automatically parses, chunks, and cleans your messy inputs into structured formats.",
    icon: FileText,
    image: "/raw-data-icon.png",
  },
  {
    id: 2,
    step: "02 - GRAPH",
    title: "Knowledge Mapping",
    description:
      "Our AI identifies entities (people, places, concepts) and their relationships, constructing a rich semantic graph from your data.",
    icon: Share2,
    image: "/knowledge-graph-icon.png",
  },
  {
    id: 3,
    step: "03 - INDEX",
    title: "Hybrid Indexing",
    description:
      "Data is stored in both vector stores (for similarity) and graph databases (for structure), ensuring comprehensive retrieval.",
    icon: Database,
    image: "/hybrid-indexing-icon.png",
  },
  {
    id: 4,
    step: "04 - RETRIEVE",
    title: "Contextual Answers",
    description:
      "Ask complex questions. The system traverses the graph to find connected facts and generates precise, sourced answers in milliseconds.",
    icon: Search,
    image: "/contextual-answers-icon.png",
  },
];

const PipelineSteps = () => {
  return (
    <section
      className="pipeline-section"
      style={{ background: "#000000", position: "relative" }}
    >
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "none",
          zIndex: 0,
        }}
      >
        <div
          style={{
            position: "sticky",
            top: "10%",
            left: 0,
            width: "100%",
            height: "80vh",
            background:
              "radial-gradient(ellipse at center, rgba(37, 99, 235, 0.6) 0%, transparent 70%)",
            filter: "blur(100px)",
          }}
        />
      </div>

      <div
        className="pipeline-container"
        style={{ position: "relative", zIndex: 1 }}
      >
        <div className="pipeline-left">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <div className="pipeline-pill">How it works</div>
            <h2 className="pipeline-heading">
              From raw data to <br />
              unforgettable <br />
              memory.
            </h2>
            <p className="pipeline-subheading">
              GraphRAG turns messy, scattered inputs into clean, contextual
              memory your AI can instantly recall — powering fast, reliable, and
              enterprise-ready products.
            </p>

            <a href="/docs" className="pipeline-cta">
              How GraphRAG works{" "}
              <ArrowRight size={18} style={{ marginLeft: "8px" }} />
            </a>
          </motion.div>
        </div>

        <div className="pipeline-right">
          {steps.map((item, index) => (
            <PipelineCard key={item.id} item={item} index={index} />
          ))}
        </div>
      </div>
    </section>
  );
};

const PipelineCard = ({
  item,
  index,
}: {
  item: (typeof steps)[0];
  index: number;
}) => {
  const getAnimationProps = (id: number) => {
    if (id === 1) {
      return {
        animate: { y: [-10, 10, -10] },
        transition: {
          repeat: Infinity,
          duration: 5,
          ease: "easeInOut" as const,
        },
      };
    } else if (id === 2) {
      return {
        animate: { scale: [1, 1.15, 1], rotate: [-5, 5, -5] },
        transition: {
          repeat: Infinity,
          duration: 4,
          ease: "easeInOut" as const,
        },
      };
    } else if (id === 4) {
      return {
        animate: { y: [-8, 8, -8], scale: [1, 1.02, 1] },
        transition: {
          repeat: Infinity,
          duration: 4.5,
          ease: "easeInOut" as const,
        },
      };
    }
    return {};
  };

  const animationProps = getAnimationProps(item.id);

  return (
    <motion.div
      className="pipeline-card"
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: index * 0.1 }}
      viewport={{ once: true, margin: "-100px" }}
    >
      <div className="card-step-number">{item.step}</div>

      <div
        className="card-visual"
        style={{
          opacity: item.image ? 1 : 0.5,
          background: item.image ? "none" : undefined,
        }}
      >
        {item.image ? (
          <div
            style={{
              position: "relative",
              width: "100%",
              height: "100%",
              display: "flex",
              alignItems: "flex-start",
              justifyContent: "center",
              paddingTop: "2rem",
            }}
          >
            <div
              style={{
                position: "absolute",
                width: "200px",
                height: "200px",
                background:
                  "radial-gradient(circle, rgba(59,130,246,0.15), transparent 70%)",
                borderRadius: "50%",
                filter: "blur(40px)",
                zIndex: 0,
                top: "10%",
              }}
            ></div>

            {item.id === 2 ? (
              <div
                style={{
                  position: "relative",
                  width: "80%",
                  height: "50%",
                  zIndex: 1,
                }}
              >
                <motion.div
                  style={{
                    position: "absolute",
                    inset: 0,
                    WebkitMaskImage:
                      "radial-gradient(circle, transparent 25%, black 65%)",
                    maskImage:
                      "radial-gradient(circle, transparent 25%, black 65%)",
                  }}
                  {...animationProps}
                >
                  <Image
                    src={item.image}
                    alt={item.title}
                    fill
                    style={{ objectFit: "contain" }}
                  />
                </motion.div>

                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    WebkitMaskImage:
                      "radial-gradient(circle, black 40%, transparent 60%)",
                    maskImage:
                      "radial-gradient(circle, black 40%, transparent 60%)",
                  }}
                >
                  <Image
                    src={item.image}
                    alt={item.title}
                    fill
                    style={{ objectFit: "contain" }}
                  />
                </div>
              </div>
            ) : item.id === 3 ? (
              <div
                style={{
                  position: "relative",
                  width: "100%",
                  height: "80%",
                  zIndex: 1,
                }}
              >
                {[
                  { t: "2%", l: "10%", s: 12, c: "#93c5fd", d: 0.1 },
                  { t: "15%", l: "75%", s: 15, c: "#2563eb", d: 0.5 },
                  { t: "5%", l: "85%", s: 20, c: "#1d4ed8", d: 3.1 },
                  { t: "25%", l: "15%", s: 14, c: "#bfdbfe", d: 1.0 },
                  { t: "20%", l: "80%", s: 12, c: "#60a5fa", d: 0.3 },
                  { t: "35%", l: "30%", s: 10, c: "#93c5fd", d: 1.2 },
                  { t: "45%", l: "50%", s: 24, c: "#1d4ed8", d: 0.8 },
                  { t: "35%", l: "70%", s: 15, c: "#3b82f6", d: 2.7 },
                  { t: "50%", l: "85%", s: 13, c: "#60a5fa", d: 1.4 },
                  { t: "45%", l: "5%", s: 16, c: "#2563eb", d: 0.6 },
                  { t: "55%", l: "45%", s: 19, c: "#1d4ed8", d: 1.9 },
                  { t: "8%", l: "5%", s: 9, c: "#3b82f6", d: 1.7 },
                  { t: "12%", l: "95%", s: 11, c: "#60a5fa", d: 2.8 },
                  { t: "70%", l: "15%", s: 20, c: "#1d4ed8", d: 1.5 },
                  { t: "75%", l: "80%", s: 15, c: "#3b82f6", d: 2.5 },
                  { t: "65%", l: "50%", s: 12, c: "#60a5fa", d: 3.0 },
                ].map((orb, i) => (
                  <motion.div
                    key={`single-${i}`}
                    style={{
                      position: "absolute",
                      top: orb.t,
                      left: orb.l,
                      width: orb.s,
                      height: orb.s,
                      backgroundColor: orb.c,
                      borderRadius: "50%",
                      boxShadow: `0 0 ${orb.s}px ${orb.c}`,
                    }}
                    animate={{
                      y: [-15, 15, -15],
                      opacity: [0.6, 1, 0.6],
                    }}
                    transition={{
                      repeat: Infinity,
                      duration: 3 + (i % 4),
                      ease: "easeInOut",
                      delay: orb.d,
                    }}
                  />
                ))}

                {[
                  {
                    t: "10%",
                    l: "55%",
                    d: 0.5,
                    nodes: [
                      { x: 0, y: 0, s: 14, c: "#3b82f6" },
                      { x: 60, y: 40, s: 18, c: "#60a5fa" },
                    ],
                  },
                  {
                    t: "30%",
                    l: "25%",
                    d: 1.2,
                    nodes: [
                      { x: 10, y: 0, s: 12, c: "#2563eb" },
                      { x: 80, y: 10, s: 10, c: "#93c5fd" },
                      { x: 40, y: 60, s: 14, c: "#1d4ed8" },
                    ],
                  },
                  {
                    t: "25%",
                    l: "50%",
                    d: 2.0,
                    nodes: [
                      { x: 0, y: 50, s: 16, c: "#bfdbfe" },
                      { x: 50, y: 0, s: 12, c: "#1e40af" },
                    ],
                  },
                  {
                    t: "45%",
                    l: "10%",
                    d: 2.8,
                    nodes: [
                      { x: 0, y: 20, s: 10, c: "#2563eb" },
                      { x: 50, y: 0, s: 14, c: "#60a5fa" },
                      { x: 90, y: 30, s: 12, c: "#93c5fd" },
                    ],
                  },
                ].map((cluster, i) => (
                  <motion.div
                    key={`cluster-${i}`}
                    style={{
                      position: "absolute",
                      top: cluster.t,
                      left: cluster.l,
                      width: 100,
                      height: 100,
                    }}
                    animate={{
                      y: [-15, 15, -15],
                      opacity: [0.8, 1, 0.8],
                    }}
                    transition={{
                      repeat: Infinity,
                      duration: 5 + (i % 2),
                      ease: "easeInOut",
                      delay: cluster.d,
                    }}
                  >
                    <svg
                      style={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        overflow: "visible",
                        width: "100%",
                        height: "100%",
                      }}
                    >
                      {cluster.nodes.map((node, ni) => {
                        if (ni === 0) return null;
                        const prev = cluster.nodes[ni - 1];
                        return (
                          <line
                            key={`line-${ni}`}
                            x1={prev.x + prev.s / 2}
                            y1={prev.y + prev.s / 2}
                            x2={node.x + node.s / 2}
                            y2={node.y + node.s / 2}
                            stroke={node.c}
                            strokeWidth="1"
                            strokeOpacity="0.5"
                          />
                        );
                      })}
                    </svg>
                    {cluster.nodes.map((node, ni) => (
                      <div
                        key={`node-${ni}`}
                        style={{
                          position: "absolute",
                          left: node.x,
                          top: node.y,
                          width: node.s,
                          height: node.s,
                          backgroundColor: node.c,
                          borderRadius: "50%",
                          boxShadow: `0 0 ${node.s}px ${node.c}`,
                        }}
                      />
                    ))}
                  </motion.div>
                ))}
              </div>
            ) : (
              <motion.div
                style={{
                  position: "relative",
                  width: "80%",
                  height: "50%",
                  zIndex: 1,
                }}
                {...animationProps}
              >
                <Image
                  src={item.image}
                  alt={item.title}
                  fill
                  style={{ objectFit: "contain" }}
                />
              </motion.div>
            )}
          </div>
        ) : (
          <div
            style={{
              position: "absolute",
              top: "40%",
              left: "60%",
              width: "200px",
              height: "200px",
              background:
                "radial-gradient(circle, rgba(59,130,246,0.2), transparent 70%)",
              borderRadius: "50%",
              filter: "blur(40px)",
            }}
          ></div>
        )}
      </div>

      <h3 className="card-title">{item.title}</h3>
      <p className="card-description">{item.description}</p>
    </motion.div>
  );
};

export default PipelineSteps;
