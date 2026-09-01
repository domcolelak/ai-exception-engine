import { api } from "@/lib/api";
import { dateTime } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function PoliciesPage() {
  const policies = await api.policies();

  if (!policies || policies.length === 0) {
    return (
      <>
        <h1>Suppression policies</h1>
        <div className="empty">
          No policies. A policy silences a pattern somebody has reviewed and accepted —
          but only below a score ceiling, so it can never hide a worse version of the
          same shape.
        </div>
      </>
    );
  }

  return (
    <>
      <h1>Suppression policies</h1>
      <p className="subtitle">
        Each policy carries a score ceiling. Above it the exception is raised anyway:
        accepting &ldquo;these are fine&rdquo; must not become a blind spot.
      </p>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Policy</th>
              <th>Matches</th>
              <th>Detectors</th>
              <th>Ceiling</th>
              <th>State</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {policies.map((policy) => (
              <tr key={policy.id}>
                <td>
                  <strong>{policy.name}</strong>
                  {policy.reason && (
                    <div className="muted" style={{ fontSize: 12 }}>
                      {policy.reason}
                    </div>
                  )}
                </td>
                <td style={{ fontSize: 12 }}>
                  {Object.entries(policy.match).length === 0 ? (
                    <span className="muted">everything</span>
                  ) : (
                    Object.entries(policy.match).map(([k, v]) => (
                      <span className="pill" key={k}>
                        {k} = {String(v)}
                      </span>
                    ))
                  )}
                </td>
                <td style={{ fontSize: 12 }}>
                  {policy.detectors.length === 0 ? (
                    <span className="muted">any</span>
                  ) : (
                    policy.detectors.map((d) => (
                      <span className="pill" key={d}>
                        {d}
                      </span>
                    ))
                  )}
                </td>
                <td style={{ fontSize: 13 }}>{policy.max_score}</td>
                <td>
                  <span className={policy.active ? "badge badge-medium" : "badge badge-low"}>
                    {policy.active ? "active" : "inactive"}
                  </span>
                </td>
                <td className="muted" style={{ fontSize: 12 }}>
                  {dateTime(policy.created_at)}
                  {policy.created_by && <div>by {policy.created_by}</div>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
