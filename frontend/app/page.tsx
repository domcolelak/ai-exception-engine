import Link from "next/link";
import { api } from "@/lib/api";
import { count, percent, severityClass } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function OverviewPage() {
  const overview = await api.overview();

  if (!overview) {
    return (
      <>
        <h1>Overview</h1>
        <div className="empty">
          <p>The API is not reachable.</p>
          <p className="muted">
            Start the backend with <code>docker compose up</code>, or run{" "}
            <code>uvicorn app.main:app --reload</code> inside <code>backend/</code>.
          </p>
        </div>
      </>
    );
  }

  return (
    <>
      <h1>Overview</h1>
      <p className="subtitle">
        {count(overview.event_count)} events analysed across {overview.schema_count}{" "}
        schema(s).
      </p>

      <div className="grid">
        <div className="card">
          <div className="card-label">Waiting for review</div>
          <div className="card-value">{overview.open_exceptions}</div>
          <div className="card-note">
            {overview.critical_exceptions} critical
          </div>
        </div>
        <div className="card">
          <div className="card-label">Resolved</div>
          <div className="card-value">{overview.resolved_exceptions}</div>
          <div className="card-note">closed by a human</div>
        </div>
        <div className="card">
          <div className="card-label">Reviewed with a label</div>
          <div className="card-value">{overview.labelled_exceptions}</div>
          <div className="card-note">feeds the retraining proposal</div>
        </div>
        <div className="card">
          <div className="card-label">Events analysed</div>
          <div className="card-value">{count(overview.event_count)}</div>
          <div className="card-note">
            {overview.event_count > 0
              ? `${percent(overview.open_exceptions / overview.event_count, 2)} raised a case`
              : "none yet"}
          </div>
        </div>
      </div>

      {overview.detector_precision.length > 0 && (
        <>
          <h2>Detector precision, from human labels</h2>
          <p className="muted" style={{ marginTop: -6, fontSize: 13 }}>
            Measured, not assumed. A detector with poor precision is a candidate for a
            lower weight — but nothing changes until somebody promotes a proposal.
          </p>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Detector</th>
                  <th>Reviewed</th>
                  <th>Confirmed</th>
                  <th>Rejected</th>
                  <th>Precision</th>
                </tr>
              </thead>
              <tbody>
                {overview.detector_precision.map((entry) => (
                  <tr key={entry.detector}>
                    <td>
                      <code>{entry.detector}</code>
                    </td>
                    <td>{entry.total}</td>
                    <td>{entry.confirmed}</td>
                    <td>{entry.rejected}</td>
                    <td>
                      <div className="row" style={{ gap: 8, flexWrap: "nowrap" }}>
                        <div className="bar" style={{ minWidth: 70 }}>
                          <span style={{ width: `${entry.precision * 100}%` }} />
                        </div>
                        <span style={{ fontSize: 13 }}>{percent(entry.precision)}</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      <h2>Highest-scoring open cases</h2>
      {overview.top_exceptions.length === 0 ? (
        <div className="empty">
          Nothing is waiting. Either behaviour is normal, or no detection run has
          happened yet.
        </div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Case</th>
                <th>Entity</th>
                <th>Score</th>
                <th>Severity</th>
                <th>Compared against</th>
              </tr>
            </thead>
            <tbody>
              {overview.top_exceptions.map((item) => (
                <tr key={item.id}>
                  <td>
                    <Link href={`/exceptions/${item.id}`}>{item.title}</Link>
                  </td>
                  <td>
                    <code>{item.entity_id}</code>
                  </td>
                  <td>{item.anomaly_score.toFixed(1)}</td>
                  <td>
                    <span className={severityClass(item.severity)}>{item.severity}</span>
                  </td>
                  <td className="muted" style={{ fontSize: 12 }}>
                    {item.baseline_scope || "all data"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p style={{ marginTop: 20 }}>
        <Link href="/exceptions" className="pill">
          Open the inbox
        </Link>
      </p>
    </>
  );
}
