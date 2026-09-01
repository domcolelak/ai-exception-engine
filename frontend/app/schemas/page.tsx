import { api } from "@/lib/api";
import { count, dateTime } from "@/lib/format";

export const dynamic = "force-dynamic";

export default async function SchemasPage() {
  const schemas = await api.schemas();

  if (!schemas || schemas.length === 0) {
    return (
      <>
        <h1>Schemas &amp; baselines</h1>
        <div className="empty">
          No schema registered. A schema tells the engine which columns are numeric,
          which are categorical, and which define context.
        </div>
      </>
    );
  }

  return (
    <>
      <h1>Schemas &amp; baselines</h1>
      <p className="subtitle">
        Context dimensions are what make a baseline contextual. They are ordered: the
        last one is the first to be dropped when a scope has too little data to judge.
      </p>

      {schemas.map((schema) => (
        <div className="card" key={schema.id} style={{ marginBottom: 14 }}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h3>{schema.name}</h3>
            <div className="row">
              <span className="pill">{count(schema.event_count)} events</span>
              <span className="pill">{schema.open_exceptions} open</span>
            </div>
          </div>

          <table style={{ marginTop: 10 }}>
            <tbody>
              <tr>
                <td style={{ width: 180, color: "var(--muted)", fontSize: 13 }}>
                  Entity
                </td>
                <td style={{ fontSize: 13 }}>
                  <code>{schema.entity_type}</code> keyed by{" "}
                  <code>{schema.entity_key}</code>
                </td>
              </tr>
              <tr>
                <td style={{ color: "var(--muted)", fontSize: 13 }}>Context dimensions</td>
                <td>
                  {schema.context_dimensions.length === 0 ? (
                    <span className="muted" style={{ fontSize: 13 }}>
                      none — every comparison is global
                    </span>
                  ) : (
                    schema.context_dimensions.map((dim, index) => (
                      <span className="pill" key={dim}>
                        {index + 1}. {dim}
                      </span>
                    ))
                  )}
                </td>
              </tr>
              <tr>
                <td style={{ color: "var(--muted)", fontSize: 13 }}>Numeric features</td>
                <td>
                  {schema.numeric_features.map((f) => (
                    <span className="pill" key={f}>
                      {f}
                    </span>
                  ))}
                </td>
              </tr>
              <tr>
                <td style={{ color: "var(--muted)", fontSize: 13 }}>
                  Categorical features
                </td>
                <td>
                  {schema.categorical_features.map((f) => (
                    <span className="pill" key={f}>
                      {f}
                    </span>
                  ))}
                </td>
              </tr>
              {schema.rate_features.length > 0 && (
                <tr>
                  <td style={{ color: "var(--muted)", fontSize: 13 }}>Rate features</td>
                  <td>
                    {schema.rate_features.map((f) => (
                      <span className="pill" key={f}>
                        {f}
                      </span>
                    ))}
                    <div className="muted" style={{ fontSize: 12 }}>
                      the share of events where these are non-zero is compared across peers
                    </div>
                  </td>
                </tr>
              )}
              <tr>
                <td style={{ color: "var(--muted)", fontSize: 13 }}>Score threshold</td>
                <td style={{ fontSize: 13 }}>{schema.min_score}</td>
              </tr>
              <tr>
                <td style={{ color: "var(--muted)", fontSize: 13 }}>Detector weights</td>
                <td style={{ fontSize: 13 }}>
                  {Object.keys(schema.detector_weights).length === 0
                    ? "defaults"
                    : Object.entries(schema.detector_weights)
                        .map(([k, v]) => `${k} ${v.toFixed(2)}`)
                        .join(" · ")}
                </td>
              </tr>
              <tr>
                <td style={{ color: "var(--muted)", fontSize: 13 }}>Registered</td>
                <td className="muted" style={{ fontSize: 13 }}>
                  {dateTime(schema.created_at)}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      ))}
    </>
  );
}
