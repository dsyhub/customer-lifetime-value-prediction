import { useState, useMemo } from "react";

const COLORS = {
  "High Value": "#2563EB",
  Growing: "#16A34A",
  "At-Risk": "#EA580C",
  "Low Value": "#9CA3AF",
};

const SEGMENTS = [
  {
    name: "High Value",
    customers: 984,
    pct: "20%",
    avgCLV: 4842,
    avgP: 0.85,
    revShare: 60.8,
    totalRev: 4762728,
    action: "Loyalty retention",
    budget: 5,
  },
  {
    name: "Growing",
    customers: 1821,
    pct: "37%",
    avgCLV: 1313,
    avgP: 0.64,
    revShare: 30.5,
    totalRev: 2391073,
    action: "Personalized offers",
    budget: 15,
  },
  {
    name: "At-Risk",
    customers: 574,
    pct: "12%",
    avgCLV: 240,
    avgP: 0.11,
    revShare: 1.8,
    totalRev: 137760,
    action: "Win-back campaign",
    budget: 10,
  },
  {
    name: "Low Value",
    customers: 1539,
    pct: "31%",
    avgCLV: 362,
    avgP: 0.37,
    revShare: 7.1,
    totalRev: 557118,
    action: "Email only",
    budget: 2,
  },
];

const SAMPLE_CUSTOMERS = [
  {
    id: "12347",
    frequency: 3,
    recency: 62,
    T: 220,
    monetary: 607.81,
    avgOrder: 608.74,
    uniqueProducts: 90,
    cancellationRate: 0.0,
    pPurchase: 0.82,
    spendTier: "Mid Spend",
    clv12m: 1392,
    segment: "High Value",
    topDrivers: [
      { feature: "recency", impact: +0.85, direction: "low recency → recent buyer" },
      { feature: "frequency", impact: +0.62, direction: "3 repeat purchases" },
      { feature: "monetary_value", impact: +0.31, direction: "above-average spend" },
      { feature: "cancellation_rate", impact: +0.18, direction: "zero cancellations" },
      { feature: "recency_ratio", impact: -0.05, direction: "moderate dormancy" },
    ],
  },
  {
    id: "12920",
    frequency: 0,
    recency: 287,
    T: 287,
    monetary: 340.0,
    avgOrder: 340.0,
    uniqueProducts: 4,
    cancellationRate: 0.0,
    pPurchase: 0.06,
    spendTier: "Mid Spend",
    clv12m: 102,
    segment: "At-Risk",
    topDrivers: [
      { feature: "recency", impact: -1.42, direction: "287 days since last purchase" },
      { feature: "frequency", impact: -0.91, direction: "zero repeat purchases" },
      { feature: "recency_ratio", impact: -0.73, direction: "dormant entire lifetime" },
      { feature: "unique_products", impact: -0.22, direction: "only 4 products" },
      { feature: "monetary_value", impact: +0.08, direction: "moderate spend" },
    ],
  },
  {
    id: "14527",
    frequency: 8,
    recency: 14,
    T: 410,
    monetary: 289.5,
    avgOrder: 312.1,
    uniqueProducts: 47,
    cancellationRate: 0.12,
    pPurchase: 0.91,
    spendTier: "Low Spend",
    clv12m: 731,
    segment: "Growing",
    topDrivers: [
      { feature: "recency", impact: +1.21, direction: "purchased 14 days ago" },
      { feature: "frequency", impact: +0.94, direction: "8 repeat purchases" },
      { feature: "cancellation_rate", impact: +0.35, direction: "active order adjuster" },
      { feature: "unique_products", impact: +0.28, direction: "broad catalog engagement" },
      { feature: "monetary_value", impact: -0.15, direction: "below-average spend per order" },
    ],
  },
];

function KPICard({ label, value, subtitle, accent }) {
  return (
    <div
      style={{
        background: "#FFFFFF",
        borderRadius: 12,
        padding: "20px 24px",
        borderLeft: `4px solid ${accent || "#2563EB"}`,
        boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
        minWidth: 0,
      }}
    >
      <div style={{ fontSize: 13, color: "#6B7280", fontWeight: 500, letterSpacing: 0.3, marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, color: "#111827", lineHeight: 1.2 }}>{value}</div>
      {subtitle && <div style={{ fontSize: 12, color: "#9CA3AF", marginTop: 4 }}>{subtitle}</div>}
    </div>
  );
}

function SegmentBar({ segments }) {
  const total = segments.reduce((s, seg) => s + seg.customers, 0);
  return (
    <div style={{ display: "flex", borderRadius: 8, overflow: "hidden", height: 36 }}>
      {segments.map((seg) => (
        <div
          key={seg.name}
          style={{
            width: `${(seg.customers / total) * 100}%`,
            background: COLORS[seg.name],
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "#fff",
            fontSize: 11,
            fontWeight: 600,
            letterSpacing: 0.2,
            minWidth: 40,
          }}
        >
          {seg.pct}
        </div>
      ))}
    </div>
  );
}

function RevenueChart({ segments }) {
  const maxRev = Math.max(...segments.map((s) => s.totalRev));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {segments.map((seg) => {
        const widthPct = (seg.totalRev / maxRev) * 100;
        const isNarrow = widthPct < 25;
        const label = seg.totalRev >= 1e6
          ? `$${(seg.totalRev / 1e6).toFixed(1)}M (${seg.revShare}%)`
          : `$${(seg.totalRev / 1e3).toFixed(0)}K (${seg.revShare}%)`;
        return (
          <div key={seg.name} style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ width: 90, fontSize: 13, fontWeight: 500, color: "#374151", textAlign: "right" }}>
              {seg.name}
            </div>
            <div style={{ flex: 1, position: "relative", height: 28, background: "#F3F4F6", borderRadius: 6 }}>
              <div
                style={{
                  width: `${widthPct}%`,
                  height: "100%",
                  background: COLORS[seg.name],
                  borderRadius: 6,
                  display: "flex",
                  alignItems: "center",
                  paddingLeft: isNarrow ? 0 : 8,
                }}
              >
                {!isNarrow && (
                  <span style={{ color: "#fff", fontSize: 12, fontWeight: 600, whiteSpace: "nowrap" }}>
                    {label}
                  </span>
                )}
              </div>
              {isNarrow && (
                <span
                  style={{
                    position: "absolute",
                    left: `calc(${widthPct}% + 8px)`,
                    top: "50%",
                    transform: "translateY(-50%)",
                    fontSize: 12,
                    fontWeight: 600,
                    color: "#374151",
                    whiteSpace: "nowrap",
                  }}
                >
                  {label}
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Tab1() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
        <KPICard label="TOTAL CUSTOMERS" value="4,918" subtitle="UCI Online Retail II" accent="#2563EB" />
        <KPICard label="PREDICTED 12M REVENUE" value="$7.8M" subtitle="CLV = P(purchase) × E[rev]" accent="#16A34A" />
        <KPICard label="TOP 20% CAPTURE" value="70.2%" subtitle="of actual holdout revenue" accent="#8B5CF6" />
        <KPICard label="BRIER SCORE" value="0.168" subtitle="vs 0.250 naive baseline" accent="#EA580C" />
      </div>

      <div
        style={{
          background: "#fff",
          borderRadius: 12,
          padding: 24,
          boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
        }}
      >
        <h3 style={{ fontSize: 15, fontWeight: 600, color: "#111827", margin: "0 0 6px 0" }}>
          Pipeline Overview
        </h3>
        <p style={{ fontSize: 13, color: "#6B7280", lineHeight: 1.6, margin: "0 0 20px 0" }}>
          Stage 1 predicts purchase probability (calibrated LightGBM). Stage 2 estimates expected revenue by
          spend tier. CLV = P(purchase) × E[revenue | purchase]. Four segments receive differentiated
          marketing strategies.
        </p>

        <div style={{ display: "flex", gap: 8, marginBottom: 20, flexWrap: "wrap" }}>
          {SEGMENTS.map((seg) => (
            <div
              key={seg.name}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                background: "#F9FAFB",
                borderRadius: 6,
                padding: "6px 12px",
              }}
            >
              <div
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: "50%",
                  background: COLORS[seg.name],
                }}
              />
              <span style={{ fontSize: 12, fontWeight: 500, color: "#374151" }}>
                {seg.name} — {seg.customers.toLocaleString()} ({seg.pct})
              </span>
            </div>
          ))}
        </div>

        <SegmentBar segments={SEGMENTS} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        <div
          style={{
            background: "#fff",
            borderRadius: 12,
            padding: 24,
            boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
          }}
        >
          <h3 style={{ fontSize: 15, fontWeight: 600, color: "#111827", margin: "0 0 16px 0" }}>
            Predicted 12-Month Revenue by Segment
          </h3>
          <RevenueChart segments={SEGMENTS} />
        </div>

        <div
          style={{
            background: "#fff",
            borderRadius: 12,
            padding: 24,
            boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
          }}
        >
          <h3 style={{ fontSize: 15, fontWeight: 600, color: "#111827", margin: "0 0 16px 0" }}>
            Segment Profiles
          </h3>
          <table style={{ width: "100%", fontSize: 13, borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ borderBottom: "2px solid #E5E7EB" }}>
                {["Segment", "Avg CLV", "P(purchase)", "Action"].map((h) => (
                  <th
                    key={h}
                    style={{
                      textAlign: "left",
                      padding: "8px 6px",
                      fontWeight: 600,
                      color: "#6B7280",
                      fontSize: 11,
                      textTransform: "uppercase",
                      letterSpacing: 0.5,
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {SEGMENTS.map((seg) => (
                <tr key={seg.name} style={{ borderBottom: "1px solid #F3F4F6" }}>
                  <td style={{ padding: "10px 6px", fontWeight: 500, color: "#111827" }}>
                    <span
                      style={{
                        display: "inline-block",
                        width: 8,
                        height: 8,
                        borderRadius: "50%",
                        background: COLORS[seg.name],
                        marginRight: 8,
                      }}
                    />
                    {seg.name}
                  </td>
                  <td style={{ padding: "10px 6px", color: "#374151" }}>
                    ${seg.avgCLV.toLocaleString()}
                  </td>
                  <td style={{ padding: "10px 6px", color: "#374151" }}>{(seg.avgP * 100).toFixed(0)}%</td>
                  <td style={{ padding: "10px 6px", color: "#6B7280", fontSize: 12 }}>{seg.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function Tab2() {
  const [selectedId, setSelectedId] = useState("12347");
  const customer = SAMPLE_CUSTOMERS.find((c) => c.id === selectedId);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div
        style={{
          background: "#fff",
          borderRadius: 12,
          padding: 24,
          boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
        }}
      >
        <label style={{ fontSize: 13, fontWeight: 500, color: "#374151", display: "block", marginBottom: 8 }}>
          Select Customer ID
        </label>
        <select
          value={selectedId}
          onChange={(e) => setSelectedId(e.target.value)}
          style={{
            padding: "10px 14px",
            borderRadius: 8,
            border: "1px solid #D1D5DB",
            fontSize: 14,
            width: 240,
            background: "#F9FAFB",
            cursor: "pointer",
          }}
        >
          {SAMPLE_CUSTOMERS.map((c) => (
            <option key={c.id} value={c.id}>
              Customer {c.id}
            </option>
          ))}
        </select>
      </div>

      {customer && (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16 }}>
            <KPICard
              label="PURCHASE PROBABILITY"
              value={`${(customer.pPurchase * 100).toFixed(0)}%`}
              accent={COLORS[customer.segment]}
            />
            <KPICard
              label="PREDICTED CLV (12M)"
              value={`$${customer.clv12m.toLocaleString()}`}
              accent={COLORS[customer.segment]}
            />
            <KPICard label="SEGMENT" value={customer.segment} accent={COLORS[customer.segment]} />
            <KPICard label="SPEND TIER" value={customer.spendTier} accent="#6B7280" />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
            <div
              style={{
                background: "#fff",
                borderRadius: 12,
                padding: 24,
                boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
              }}
            >
              <h3 style={{ fontSize: 15, fontWeight: 600, color: "#111827", margin: "0 0 16px 0" }}>
                Customer Profile
              </h3>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                {[
                  ["Frequency", customer.frequency],
                  ["Recency", `${customer.recency} days`],
                  ["Tenure", `${customer.T} days`],
                  ["Monetary Value", `$${customer.monetary.toFixed(0)}`],
                  ["Avg Order Value", `$${customer.avgOrder.toFixed(0)}`],
                  ["Unique Products", customer.uniqueProducts],
                  ["Cancellation Rate", `${(customer.cancellationRate * 100).toFixed(0)}%`],
                ].map(([label, val]) => (
                  <div key={label} style={{ display: "flex", justifyContent: "space-between", padding: "8px 0", borderBottom: "1px solid #F3F4F6" }}>
                    <span style={{ fontSize: 13, color: "#6B7280" }}>{label}</span>
                    <span style={{ fontSize: 13, fontWeight: 600, color: "#111827" }}>{val}</span>
                  </div>
                ))}
              </div>
            </div>

            <div
              style={{
                background: "#fff",
                borderRadius: 12,
                padding: 24,
                boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
              }}
            >
              <h3 style={{ fontSize: 15, fontWeight: 600, color: "#111827", margin: "0 0 16px 0" }}>
                Prediction Drivers (SHAP)
              </h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {customer.topDrivers.map((d) => {
                  const maxAbs = Math.max(...customer.topDrivers.map((x) => Math.abs(x.impact)));
                  const pct = (Math.abs(d.impact) / maxAbs) * 100;
                  const isPositive = d.impact > 0;
                  return (
                    <div key={d.feature}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
                        <span style={{ fontSize: 12, fontWeight: 600, color: "#374151", fontFamily: "monospace" }}>
                          {d.feature}
                        </span>
                        <span
                          style={{
                            fontSize: 12,
                            fontWeight: 600,
                            color: isPositive ? "#16A34A" : "#DC2626",
                          }}
                        >
                          {isPositive ? "+" : ""}
                          {d.impact.toFixed(2)}
                        </span>
                      </div>
                      <div style={{ height: 6, background: "#F3F4F6", borderRadius: 3, overflow: "hidden" }}>
                        <div
                          style={{
                            width: `${pct}%`,
                            height: "100%",
                            background: isPositive ? "#16A34A" : "#DC2626",
                            borderRadius: 3,
                            transition: "width 0.3s ease",
                          }}
                        />
                      </div>
                      <div style={{ fontSize: 11, color: "#9CA3AF", marginTop: 2 }}>{d.direction}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function Tab3() {
  const [budgets, setBudgets] = useState({
    "High Value": 5,
    Growing: 15,
    "At-Risk": 10,
    "Low Value": 2,
  });

  const liftRange = useMemo(() => {
    const arr = [];
    for (let i = 0.5; i <= 20; i += 0.5) arr.push(i);
    return arr;
  }, []);

  const breakeven = SEGMENTS.map((seg) => {
    const b = budgets[seg.name];
    const be = b / seg.avgCLV;
    const oneInN = Math.round(1 / be);
    return { ...seg, budget: b, beLift: be, oneInN };
  });

  const chartMax = useMemo(() => {
    let max = 0;
    SEGMENTS.forEach((seg) => {
      const b = budgets[seg.name];
      const roi = (20 / 100) * seg.avgCLV * seg.customers - b * seg.customers;
      if (roi > max) max = roi;
    });
    return max;
  }, [budgets]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <div
        style={{
          background: "#fff",
          borderRadius: 12,
          padding: 24,
          boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
        }}
      >
        <h3 style={{ fontSize: 15, fontWeight: 600, color: "#111827", margin: "0 0 4px 0" }}>
          Campaign Budget per Customer
        </h3>
        <p style={{ fontSize: 12, color: "#9CA3AF", margin: "0 0 16px 0" }}>
          Adjust sliders to see how breakeven lift and ROI change
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 20 }}>
          {SEGMENTS.map((seg) => (
            <div key={seg.name}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 13, fontWeight: 500, color: "#374151" }}>
                  <span
                    style={{
                      display: "inline-block",
                      width: 8,
                      height: 8,
                      borderRadius: "50%",
                      background: COLORS[seg.name],
                      marginRight: 6,
                    }}
                  />
                  {seg.name}
                </span>
                <span style={{ fontSize: 13, fontWeight: 700, color: "#111827" }}>
                  ${budgets[seg.name]}
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={30}
                value={budgets[seg.name]}
                onChange={(e) =>
                  setBudgets((prev) => ({ ...prev, [seg.name]: Number(e.target.value) }))
                }
                style={{ width: "100%", accentColor: COLORS[seg.name] }}
              />
            </div>
          ))}
        </div>
      </div>

      <div
        style={{
          background: "#fff",
          borderRadius: 12,
          padding: 24,
          boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
        }}
      >
        <h3 style={{ fontSize: 15, fontWeight: 600, color: "#111827", margin: "0 0 16px 0" }}>
          Break-Even Incremental Lift
        </h3>
        <table style={{ width: "100%", fontSize: 13, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "2px solid #E5E7EB" }}>
              {["Segment", "Budget", "Avg CLV", "Break-Even Lift", "Intuition", "Action"].map((h) => (
                <th
                  key={h}
                  style={{
                    textAlign: "left",
                    padding: "8px 6px",
                    fontWeight: 600,
                    color: "#6B7280",
                    fontSize: 11,
                    textTransform: "uppercase",
                    letterSpacing: 0.5,
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {breakeven.map((seg) => (
              <tr key={seg.name} style={{ borderBottom: "1px solid #F3F4F6" }}>
                <td style={{ padding: "10px 6px", fontWeight: 500 }}>
                  <span
                    style={{
                      display: "inline-block",
                      width: 8,
                      height: 8,
                      borderRadius: "50%",
                      background: COLORS[seg.name],
                      marginRight: 8,
                    }}
                  />
                  {seg.name}
                </td>
                <td style={{ padding: "10px 6px" }}>${seg.budget}</td>
                <td style={{ padding: "10px 6px" }}>${seg.avgCLV.toLocaleString()}</td>
                <td style={{ padding: "10px 6px", fontWeight: 600, color: "#111827" }}>
                  {seg.budget === 0 ? "N/A" : `${(seg.beLift * 100).toFixed(2)}%`}
                </td>
                <td style={{ padding: "10px 6px", color: "#6B7280" }}>
                  {seg.budget === 0 ? "—" : `1 in ${seg.oneInN.toLocaleString()}`}
                </td>
                <td style={{ padding: "10px 6px", color: "#6B7280", fontSize: 12 }}>{seg.action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div
        style={{
          background: "#fff",
          borderRadius: 12,
          padding: 24,
          boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
        }}
      >
        <h3 style={{ fontSize: 15, fontWeight: 600, color: "#111827", margin: "0 0 4px 0" }}>
          ROI Sensitivity Chart
        </h3>
        <p style={{ fontSize: 12, color: "#9CA3AF", margin: "0 0 16px 0" }}>
          Total Net ROI by incremental lift percentage. Dots = break-even point.
        </p>
        <div style={{ position: "relative", height: 300, borderLeft: "1px solid #E5E7EB", borderBottom: "1px solid #E5E7EB", marginLeft: 60 }}>
          {/* Y-axis labels */}
          {[0, 0.25, 0.5, 0.75, 1].map((frac) => (
            <div
              key={frac}
              style={{
                position: "absolute",
                left: -58,
                bottom: `${frac * 100}%`,
                fontSize: 11,
                color: "#9CA3AF",
                transform: "translateY(50%)",
              }}
            >
              ${((frac * chartMax) / 1e3).toFixed(0)}K
            </div>
          ))}
          {/* Grid lines */}
          {[0.25, 0.5, 0.75, 1].map((frac) => (
            <div
              key={`grid-${frac}`}
              style={{
                position: "absolute",
                left: 0,
                right: 0,
                bottom: `${frac * 100}%`,
                borderBottom: "1px dashed #F3F4F6",
              }}
            />
          ))}
          {/* X-axis labels */}
          {[0, 5, 10, 15, 20].map((v) => (
            <div
              key={`x-${v}`}
              style={{
                position: "absolute",
                bottom: -22,
                left: `${(v / 20) * 100}%`,
                fontSize: 11,
                color: "#9CA3AF",
                transform: "translateX(-50%)",
              }}
            >
              {v}%
            </div>
          ))}
          {/* Lines for each segment */}
          {SEGMENTS.map((seg) => {
            const b = budgets[seg.name];
            const points = liftRange.map((lift) => {
              const roi = (lift / 100) * seg.avgCLV * seg.customers - b * seg.customers;
              const x = (lift / 20) * 100;
              const y = Math.max(0, (roi / chartMax) * 100);
              return `${x}%,${100 - y}%`;
            });
            const beLift = b > 0 ? (b / seg.avgCLV) * 100 : 0;
            const beX = (beLift / 20) * 100;
            return (
              <div key={seg.name}>
                <svg
                  style={{ position: "absolute", inset: 0, width: "100%", height: "100%", overflow: "visible" }}
                  preserveAspectRatio="none"
                >
                  <polyline
                    points={points.map((p) => {
                      const [xPct, yPct] = p.split(",");
                      return `${parseFloat(xPct)}%,${parseFloat(yPct)}%`;
                    }).join(" ")}
                    fill="none"
                    stroke={COLORS[seg.name]}
                    strokeWidth="2.5"
                    vectorEffect="non-scaling-stroke"
                  />
                </svg>
                {b > 0 && beLift <= 20 && (
                  <div
                    style={{
                      position: "absolute",
                      left: `${beX}%`,
                      bottom: 0,
                      width: 10,
                      height: 10,
                      borderRadius: "50%",
                      background: COLORS[seg.name],
                      border: "2px solid #fff",
                      transform: "translate(-50%, 50%)",
                      boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
                    }}
                  />
                )}
              </div>
            );
          })}
        </div>
        <div style={{ textAlign: "center", fontSize: 12, color: "#9CA3AF", marginTop: 28 }}>
          Incremental Lift (percentage points)
        </div>
        <div style={{ display: "flex", gap: 16, justifyContent: "center", marginTop: 12 }}>
          {SEGMENTS.map((seg) => (
            <div key={seg.name} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div style={{ width: 12, height: 3, background: COLORS[seg.name], borderRadius: 2 }} />
              <span style={{ fontSize: 12, color: "#6B7280" }}>{seg.name}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function CLVDashboard() {
  const [tab, setTab] = useState(0);
  const tabs = ["Executive Summary", "Customer Explorer", "Campaign Sensitivity"];

  return (
    <div
      style={{
        fontFamily: "'DM Sans', 'Segoe UI', system-ui, sans-serif",
        background: "#F8FAFC",
        minHeight: "100vh",
        padding: "0 0 40px 0",
      }}
    >
      {/* Header */}
      <div
        style={{
          background: "linear-gradient(135deg, #1E293B 0%, #334155 100%)",
          padding: "28px 40px 0 40px",
        }}
      >
        <div style={{ maxWidth: 1100, margin: "0 auto" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 4 }}>
            <h1 style={{ fontSize: 22, fontWeight: 700, color: "#F8FAFC", margin: 0, letterSpacing: -0.3 }}>
              Customer Lifetime Value
            </h1>
            <span style={{ fontSize: 13, color: "#94A3B8", fontWeight: 400 }}>
              CLV Prediction & Segmentation Dashboard
            </span>
          </div>
          <p style={{ fontSize: 13, color: "#94A3B8", margin: "4px 0 20px 0" }}>
            UCI Online Retail II · 4,918 customers · Two-stage ML pipeline
          </p>
          <div style={{ display: "flex", gap: 0 }}>
            {tabs.map((t, i) => (
              <button
                key={t}
                onClick={() => setTab(i)}
                style={{
                  padding: "10px 20px",
                  fontSize: 13,
                  fontWeight: tab === i ? 600 : 400,
                  color: tab === i ? "#F8FAFC" : "#94A3B8",
                  background: "none",
                  border: "none",
                  borderBottom: tab === i ? "2px solid #3B82F6" : "2px solid transparent",
                  cursor: "pointer",
                  transition: "all 0.15s ease",
                }}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "24px 40px" }}>
        {tab === 0 && <Tab1 />}
        {tab === 1 && <Tab2 />}
        {tab === 2 && <Tab3 />}
      </div>
    </div>
  );
}
