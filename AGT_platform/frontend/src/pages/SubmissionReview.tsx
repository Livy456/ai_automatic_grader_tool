import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api";
import {
  Container,
  Typography,
  Card,
  CardContent,
  Chip,
  LinearProgress,
} from "@mui/material";

const ACTIVE = new Set(["uploading", "uploaded", "queued", "grading"]);

function statusLabel(status: string): string {
  if (status === "uploaded") return "Uploaded — waiting to queue";
  if (status === "queued") return "Queued for grading";
  if (status === "grading") return "Grading in progress";
  if (status === "graded") return "Completed";
  if (status === "needs_review") return "Completed (needs review)";
  if (status === "error") return "Failed";
  return status;
}

export default function SubmissionReview() {
  const { id } = useParams();
  const [sub, setSub] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    let cancelled = false;
    const tick = () => {
      api
        .get(`/api/submissions/${id}`)
        .then((r) => {
          if (!cancelled) setSub(r as Record<string, unknown>);
        })
        .catch(() => {});
    };
    tick();
    const interval = setInterval(tick, 2000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [id]);

  if (!sub) {
    return (
      <Container sx={{ mt: 3 }}>
        <LinearProgress />
      </Container>
    );
  }

  const status = String(sub.status ?? "");
  const busy = ACTIVE.has(status);

  return (
    <Container sx={{ mt: 3 }}>
      <Typography variant="h5">Submission #{String(sub.id)}</Typography>
      <Chip
        label={statusLabel(status)}
        color={busy ? "primary" : status === "error" ? "error" : "success"}
        sx={{ my: 2 }}
      />
      {busy && <LinearProgress sx={{ mb: 2 }} />}
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Uploading to storage is separate from grading. This page refreshes while work is in
        progress.
      </Typography>

      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="h6">Suggested Score</Typography>
          <Typography>{sub.final_score != null ? String(sub.final_score) : "—"}</Typography>
          <Typography variant="h6" sx={{ mt: 2 }}>
            Summary Feedback
          </Typography>
          <Typography>{sub.final_feedback != null ? String(sub.final_feedback) : "—"}</Typography>
        </CardContent>
      </Card>

      {Array.isArray(sub.ai_scores) &&
        sub.ai_scores.map((s: Record<string, unknown>, idx: number) => (
          <Card key={idx} sx={{ mb: 1 }}>
            <CardContent>
              <Typography variant="subtitle1">
                {String(s.criterion)} — {String(s.score)} (conf {String(s.confidence)})
              </Typography>
              <Typography>{String(s.rationale ?? "")}</Typography>
            </CardContent>
          </Card>
        ))}
    </Container>
  );
}
