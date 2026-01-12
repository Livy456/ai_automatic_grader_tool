import React, { useEffect, useMemo, useState } from "react";
import {
  getAssignment,
  listAssignments,
  startGrading,
  uploadAssignment,
  type Assignment,
} from "../api";

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

export default function AssignmentsPage() {
  const [assignments, setAssignments] = useState<Assignment[]>([]);
  const [loading, setLoading] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const busyAssignment = useMemo(
    () => (busyId ? assignments.find((a) => a.id === busyId) : null),
    [busyId, assignments]
  );

  async function refresh() {
    const data = await listAssignments();
    setAssignments(data);
  }

  useEffect(() => {
    refresh().catch((e) => setError(String(e)));
  }, []);

  async function handleUploadAndGrade() {
    if (!selectedFile) return;
    setError(null);
    setLoading(true);

    try {
      // 1) upload
      const { id } = await uploadAssignment(selectedFile);
      setBusyId(id);

      // 2) refresh list
      await refresh();

      // 3) start grading
      await startGrading(id);

      // 4) poll status until graded/failed
      for (let i = 0; i < 60; i++) {
        const a = await getAssignment(id);
        setAssignments((prev) => {
          const rest = prev.filter((x) => x.id !== id);
          return [a, ...rest];
        });

        if (a.status === "graded" || a.status === "failed") break;
        await sleep(1500);
      }
    } catch (e: any) {
      setError(e?.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ padding: 24, fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ marginBottom: 8 }}>Assignments</h1>
      <p style={{ marginTop: 0, opacity: 0.8 }}>
        Upload an assignment file and run AI-assisted grading.
      </p>

      <div style={{ display: "flex", gap: 12, alignItems: "center", margin: "16px 0" }}>
        <input
          type="file"
          onChange={(e) => setSelectedFile(e.target.files?.[0] ?? null)}
        />
        <button
          onClick={handleUploadAndGrade}
          disabled={!selectedFile || loading}
          style={{
            padding: "10px 14px",
            borderRadius: 10,
            border: "1px solid #ccc",
            cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Working..." : "Upload & Grade"}
        </button>
      </div>

      {error && (
        <div style={{ padding: 12, border: "1px solid #f99", borderRadius: 10, marginBottom: 16 }}>
          <b>Error:</b> {error}
        </div>
      )}

      {busyAssignment && (
        <div style={{ padding: 12, border: "1px solid #ddd", borderRadius: 10, marginBottom: 16 }}>
          <b>Latest job:</b> {busyAssignment.filename} — <code>{busyAssignment.status}</code>
          {busyAssignment.status === "graded" && (
            <div style={{ marginTop: 8 }}>
              <div><b>Suggested grade:</b> {busyAssignment.suggested_grade ?? "—"}</div>
              <div style={{ whiteSpace: "pre-wrap" }}>
                <b>Feedback:</b>{" "}
                {busyAssignment.feedback ?? "No feedback returned."}
              </div>
            </div>
          )}
        </div>
      )}

      <h2 style={{ marginTop: 24 }}>All submissions</h2>
      <div style={{ display: "grid", gap: 10 }}>
        {assignments.map((a) => (
          <div key={a.id} style={{ padding: 12, border: "1px solid #ddd", borderRadius: 10 }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
              <div>
                <b>{a.filename}</b>
                <div style={{ opacity: 0.8 }}>
                  Status: <code>{a.status}</code>
                </div>
              </div>
              {a.status === "graded" && (
                <div style={{ textAlign: "right" }}>
                  <div><b>{a.suggested_grade ?? "—"}</b>/100</div>
                </div>
              )}
            </div>

            {a.status === "graded" && a.feedback && (
              <div style={{ marginTop: 8, whiteSpace: "pre-wrap" }}>{a.feedback}</div>
            )}

            {a.status === "uploaded" && (
              <button
                onClick={async () => {
                  setError(null);
                  setBusyId(a.id);
                  try {
                    await startGrading(a.id);
                  } catch (e: any) {
                    setError(e?.message ?? String(e));
                  }
                }}
                style={{
                  marginTop: 10,
                  padding: "8px 12px",
                  borderRadius: 10,
                  border: "1px solid #ccc",
                }}
              >
                Grade
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
