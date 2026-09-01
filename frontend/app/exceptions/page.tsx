import Link from "next/link";
import { api } from "@/lib/api";
import { dateTime, percent, severityClass } from "@/lib/format";

export const dynamic = "force-dynamic";

const DETECTOR_LABELS: Record<string, string> = {
  robust_deviation: "Deviation",
  categorical_rarity: "Rarity",
  isolation_forest: "Combination",
  rolling_quantile: "Own history",
  change_point: "Behaviour shift",
  peer_group: "Unlike peers",
};

export default async function ExceptionsPage({
  searchParams,
}: {
  searchParams: { status?: string; severity?: string };
}) {
  const filters: Record<string, string> = {};
  if (searchParams.status) filters.status = searchParams.status;
  if (searchParams.severity) filters.severity = searchParams.severity;

  const exceptions = await api.exceptions(filters);

  if (!exceptions || exceptions.length === 0) {
    return (
      <>
        <h1>Exception inbox</h1>
        <div className="empty">
          Nothing matches. This is a work queue, not a dashboard — an empty inbox is the
          goal, not a failure.
        </div>
      </>
    );
  }

  return (
    <>
      <h1>Exception inbox</h1>
      <p className="subtitle">
        {exceptions.length} case(s), most severe first. Each one names the population it
        was measured against, so &ldquo;unusual&rdquo; always means unusual for something
        specific.
      </p>

      <div className="row" style={{ marginBottom: 16 }}>
        <Link href="/exceptions" className="pill">
          All
        </Link>
        <Link href="/exceptions?status=open" className="pill">
          Open
        </Link>
        <Link href="/exceptions?severity=critical" className="pill">
          Critical
        </Link>
        <Link href="/exceptions?severity=high" className="pill">
          High
        </Link>
      </div>

      {exceptions.map((item) => {
        const primary = item.reasons[0];
        return (
          <div className="card" key={item.id} style={{ marginBottom: 14 }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h3>
                <Link href={`/exceptions/${item.id}`}>{item.title}</Link>
              </h3>
              <div className="row">
                {primary && (
                  <span className="pill">
                    {DETECTOR_LABELS[primary.detector] ?? primary.detector}
                  </span>
                )}
                <span className="pill">score {item.anomaly_score.toFixed(1)}</span>
                <span className={severityClass(item.severity)}>{item.severity}</span>
              </div>
            </div>

            {primary && <p style={{ margin: "6px 0" }}>{primary.explanation}</p>}

            <p className="muted" style={{ fontSize: 13, marginBottom: 0 }}>
              <code>{item.entity_id}</code> · compared against{" "}
              {item.baseline_scope || "all data"} · confidence{" "}
              {percent(item.confidence)} · {dateTime(item.detected_at)} · status{" "}
              {item.status}
              {item.assignee ? ` · assigned to ${item.assignee}` : ""}
            </p>

            {item.reasons.length > 1 && (
              <p className="muted" style={{ fontSize: 12, marginBottom: 0 }}>
                +{item.reasons.length - 1} further reason(s)
              </p>
            )}
          </div>
        );
      })}
    </>
  );
}
