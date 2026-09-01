import { api } from "@/lib/api";
import { percent } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function FeedbackPage() {
  const schemas = await api.schemas();
  if (!schemas || schemas.length === 0) {
    return (
      <>
        <h1>Detector health</h1>
        <div className="empty">No schema registered yet.</div>
      </>
    );
  }

  const primary = schemas.reduce((best, current) =>
    current.event_count > best.event_count ? current : best,
  );
  const proposal = await api.retrainingProposal(primary.id);

  return (
    <>
      <h1>Detector health</h1>
      <p className="subtitle">
        What the accumulated feedback suggests for <code>{primary.name}</code>. Nothing
        on this page has been applied — a label never reconfigures detection on its own.
      </p>

      <div className="grid">
        <div className="card">
          <div className="card-label">Labelled cases</div>
          <div className="card-value">{proposal?.labelled_count ?? 0}</div>
          <div className="card-note">human verdicts on record</div>
        </div>
        <div className="card">
          <div className="card-label">Proposal state</div>
          <div className="card-value" style={{ fontSize: 20 }}>
            {proposal?.applied ? "applied" : "not applied"}
          </div>
          <div className="card-note">promotion is a separate, explicit action</div>
        </div>
        <div className="card">
          <div className="card-label">Current threshold</div>
          <div className="card-value">{primary.min_score}</div>
          <div className="card-note">
            {proposal?.suggested_min_score
              ? `suggested: ${proposal.suggested_min_score}`
              : "no change suggested"}
          </div>
        </div>
        <div className="card">
          <div className="card-label">Proposed policies</div>
          <div className="card-value">{proposal?.proposed_policies.length ?? 0}</div>
          <div className="card-note">repeated, agreed-upon non-problems</div>
        </div>
      </div>

      {proposal && proposal.detector_quality.length > 0 && (
        <>
          <h2>Measured precision per detector</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Detector</th>
                  <th>Reviewed</th>
                  <th>Confirmed</th>
                  <th>Rejected</th>
                  <th>Precision</th>
                  <th>Suggested weight</th>
                </tr>
              </thead>
              <tbody>
                {proposal.detector_quality.map((entry) => (
                  <tr key={entry.detector}>
                    <td>
                      <code>{entry.detector}</code>
                    </td>
                    <td>{entry.total}</td>
                    <td>{entry.confirmed}</td>
                    <td>{entry.rejected}</td>
                    <td>{percent(entry.precision)}</td>
                    <td>
                      {proposal.suggested_weights[entry.detector] !== undefined
                        ? proposal.suggested_weights[entry.detector].toFixed(2)
                        : "unchanged"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {proposal && proposal.proposed_policies.length > 0 && (
        <>
          <h2>Suggested suppression policies</h2>
          {proposal.proposed_policies.map((policy) => (
            <div className="card" key={policy.name} style={{ marginBottom: 12 }}>
              <h3>{policy.name}</h3>
              <p style={{ margin: "6px 0" }}>{policy.reason}</p>
              <p className="muted" style={{ fontSize: 13, marginBottom: 0 }}>
                Would apply below a score of {policy.max_score}, based on{" "}
                {policy.supporting_exceptions.length} reviewed case(s).
              </p>
            </div>
          ))}
        </>
      )}

      {proposal && proposal.notes.length > 0 && (
        <>
          <h2>Notes</h2>
          <div className="card">
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {proposal.notes.map((note) => (
                <li key={note} style={{ marginBottom: 6 }}>
                  {note}
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </>
  );
}
