import Link from "next/link";
import { notFound } from "next/navigation";
import { api } from "@/lib/api";
import { dateTime, percent, severityClass } from "@/lib/format";

export const dynamic = "force-dynamic";

const DETECTOR_LABELS: Record<string, string> = {
  robust_deviation: "Deviation from the contextual baseline",
  categorical_rarity: "Rare value for this context",
  isolation_forest: "Unusual combination of values",
  rolling_quantile: "Departure from the entity's own history",
  change_point: "Sustained shift in behaviour",
  peer_group: "Unlike comparable entities",
};

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export default async function ExceptionDetailPage({ params }: { params: { id: string } }) {
  const [detail, similar] = await Promise.all([
    api.exception(params.id),
    api.similar(params.id),
  ]);
  if (!detail) notFound();

  const detectors = detail.evidence.detectors ?? [];
  const applicable = detectors.filter((d) => d.applicable);
  const abstained = detectors.filter((d) => !d.applicable);
  const scoreMatches = Math.abs(detail.recomputed_score - detail.anomaly_score) < 0.01;

  return (
    <>
      <h1>{detail.title}</h1>
      <p className="subtitle">
        <code>{detail.entity_type} {detail.entity_id}</code> · detected{" "}
        {dateTime(detail.detected_at)} · status {detail.status}
      </p>

      <div className="grid">
        <div className="card">
          <div className="card-label">Anomaly score</div>
          <div className="card-value">{detail.anomaly_score.toFixed(1)}</div>
          <div className="card-note">
            <span className={severityClass(detail.severity)}>{detail.severity}</span>
          </div>
        </div>
        <div className="card">
          <div className="card-label">Confidence</div>
          <div className="card-value">{percent(detail.confidence)}</div>
          <div className="card-note">
            {applicable.length} of {detectors.length} detectors could judge
          </div>
        </div>
        <div className="card">
          <div className="card-label">Compared against</div>
          <div className="card-value" style={{ fontSize: 15 }}>
            {detail.baseline_scope || "all data"}
          </div>
          <div className="card-note">the population that defined &ldquo;normal&rdquo;</div>
        </div>
        <div className="card">
          <div className="card-label">Reproducible</div>
          <div className="card-value" style={{ fontSize: 18 }}>
            {scoreMatches ? "yes" : "MISMATCH"}
          </div>
          <div className="card-note">
            recomputed from stored detector output: {detail.recomputed_score.toFixed(1)}
          </div>
        </div>
      </div>

      <h2>Why this is abnormal</h2>
      {detail.reasons.length === 0 ? (
        <div className="empty">No reasons recorded.</div>
      ) : (
        detail.reasons.map((reason, index) => (
          <div className="card" key={`${reason.detector}-${reason.feature}-${index}`} style={{ marginBottom: 12 }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h3>{reason.feature}</h3>
              <span className="pill">
                {DETECTOR_LABELS[reason.detector] ?? reason.detector}
              </span>
            </div>
            <p style={{ margin: "6px 0" }}>{reason.explanation}</p>
            <table>
              <tbody>
                <tr>
                  <td style={{ width: 150, color: "var(--muted)", fontSize: 13 }}>
                    Observed
                  </td>
                  <td style={{ fontSize: 13 }}>{formatValue(reason.observed)}</td>
                </tr>
                {reason.expected_range && (
                  <tr>
                    <td style={{ color: "var(--muted)", fontSize: 13 }}>Expected range</td>
                    <td style={{ fontSize: 13 }}>
                      {formatValue(reason.expected_range[0])} to{" "}
                      {formatValue(reason.expected_range[1])}
                    </td>
                  </tr>
                )}
                <tr>
                  <td style={{ color: "var(--muted)", fontSize: 13 }}>Deviation</td>
                  <td>
                    <div className="row" style={{ gap: 8, flexWrap: "nowrap" }}>
                      <div className="bar" style={{ minWidth: 80 }}>
                        <span style={{ width: `${reason.deviation_score * 100}%` }} />
                      </div>
                      <span style={{ fontSize: 13 }}>
                        {percent(reason.deviation_score)}
                      </span>
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        ))
      )}

      <h2>How the score was reached</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Detector</th>
              <th>Score</th>
              <th>Weight</th>
              <th>Version</th>
            </tr>
          </thead>
          <tbody>
            {applicable.map((d) => (
              <tr key={d.detector}>
                <td>
                  <code>{d.detector}</code>
                  {d.detector === detail.evidence.strongest_detector && (
                    <div>
                      <span className="pill">leading</span>
                    </div>
                  )}
                </td>
                <td>{percent(d.score)}</td>
                <td>{(detail.evidence.weights?.[d.detector] ?? 1).toFixed(2)}</td>
                <td className="muted" style={{ fontSize: 12 }}>
                  {d.version}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {abstained.length > 0 && (
        <p className="muted" style={{ fontSize: 13 }}>
          {abstained.length} detector(s) abstained rather than scoring zero:{" "}
          {abstained.map((d) => (
            <span className="pill" key={d.detector}>
              {d.detector}: {String(d.evidence?.note ?? "no data")}
            </span>
          ))}
        </p>
      )}

      <h2>Similar past cases</h2>
      {!similar || similar.similar.length === 0 ? (
        <div className="empty">Nothing comparable in the history yet.</div>
      ) : (
        <>
          <p className="muted" style={{ marginTop: -6, fontSize: 13 }}>
            {similar.note}
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Case</th>
                  <th>Similarity</th>
                  <th>Outcome</th>
                </tr>
              </thead>
              <tbody>
                {similar.similar.map((match) => (
                  <tr key={match.exception_id}>
                    <td>
                      <Link href={`/exceptions/${match.exception_id}`}>{match.title}</Link>
                      <div className="muted" style={{ fontSize: 12 }}>
                        {Object.entries(match.components)
                          .filter(([, v]) => v > 0)
                          .map(([k, v]) => `${k} ${percent(v)}`)
                          .join(" · ")}
                      </div>
                    </td>
                    <td>{percent(match.score)}</td>
                    <td className="muted" style={{ fontSize: 13 }}>
                      {match.status}
                      {match.feedback_label ? ` · ${match.feedback_label}` : ""}
                      {match.resolution_note ? ` — ${match.resolution_note}` : ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {detail.feedback.length > 0 && (
        <>
          <h2>Review history</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Label</th>
                  <th>Note</th>
                  <th>By</th>
                  <th>When</th>
                </tr>
              </thead>
              <tbody>
                {detail.feedback.map((f, index) => (
                  <tr key={index}>
                    <td>
                      <span className="pill">{f.label}</span>
                    </td>
                    <td>{f.note || "—"}</td>
                    <td className="muted">{f.author || "—"}</td>
                    <td className="muted">{dateTime(f.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {Object.keys(detail.narrative).length > 0 && (
        <>
          <h2>Narrative</h2>
          <p className="muted" style={{ marginTop: -6, fontSize: 13 }}>
            Generated from the evidence above. It adds wording, never numbers.
          </p>
          <pre className="evidence">{JSON.stringify(detail.narrative, null, 2)}</pre>
        </>
      )}

      <h2>The event</h2>
      <pre className="evidence">{JSON.stringify(detail.event_payload, null, 2)}</pre>

      <p style={{ marginTop: 20 }}>
        <Link href="/exceptions" className="pill">
          Back to the inbox
        </Link>
      </p>
    </>
  );
}
