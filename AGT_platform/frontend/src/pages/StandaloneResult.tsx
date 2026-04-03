import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import ArrowBackOutlined from "@mui/icons-material/ArrowBackOutlined";
import { getStandaloneSubmission, type StandaloneSubmissionDetail } from "../api";
import StatusChip from "../components/StatusChip";

const POLL_STATUSES = new Set(["uploading", "uploaded", "queued", "grading"]);

function gradeBarColor(score: number): "success" | "warning" | "error" {
  if (score >= 90) return "success";
  if (score >= 70) return "warning";
  return "error";
}

function criterionChipColor(fraction: number): "success" | "warning" | "error" {
  if (fraction >= 0.8) return "success";
  if (fraction >= 0.6) return "warning";
  return "error";
}

export default function StandaloneResult() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [sub, setSub] = useState<StandaloneSubmissionDetail | null>(null);
  const [loadError, setLoadError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | undefined;

    const tick = async () => {
      if (!id) return;
      try {
        const s = await getStandaloneSubmission(parseInt(id, 10));
        if (cancelled) return;
        setSub(s);
        setLoadError(false);
        const st = String(s.status);
        if (!POLL_STATUSES.has(st)) {
          if (timer) clearInterval(timer);
        }
      } catch {
        if (!cancelled) setLoadError(true);
      }
    };

    void tick();
    timer = setInterval(() => void tick(), 4000);
    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [id]);

  if (loadError || !id) {
    return (
      <Box>
        <Typography color="error">Could not load this submission.</Typography>
        <Button startIcon={<ArrowBackOutlined />} onClick={() => navigate("/autograder")} sx={{ mt: 2 }}>
          Back to Autograder
        </Button>
      </Box>
    );
  }

  if (!sub) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", py: 6 }}>
        <CircularProgress aria-label="Loading submission" />
      </Box>
    );
  }

  const scores = sub.ai_scores ?? [];
  const polling = POLL_STATUSES.has(sub.status);

  return (
    <Box>
      <Button
        startIcon={<ArrowBackOutlined />}
        onClick={() => navigate("/autograder")}
        sx={{ mb: 2 }}
        aria-label="Back to autograder"
      >
        Autograder
      </Button>

      <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2, flexWrap: "wrap" }}>
        <Typography variant="h3" component="h1">
          {sub.title}
        </Typography>
        <StatusChip status={sub.status} />
      </Box>

      {polling && <LinearProgress sx={{ mb: 2 }} aria-label="Grading in progress" />}

      {sub.final_score != null && !polling && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="overline" color="text.secondary">
              Overall score
            </Typography>
            <Box sx={{ display: "flex", alignItems: "baseline", gap: 2, mt: 1 }}>
              <Chip
                label={`${sub.final_score.toFixed(1)} / 100`}
                color={gradeBarColor(Number(sub.final_score))}
                sx={{ fontSize: "1.1rem", fontWeight: 700, height: 40 }}
              />
            </Box>
            {sub.final_feedback && (
              <Typography variant="body1" sx={{ mt: 2, whiteSpace: "pre-wrap" }}>
                {sub.final_feedback}
              </Typography>
            )}
          </CardContent>
        </Card>
      )}

      <Typography variant="h3" sx={{ mb: 1 }}>
        Criteria
      </Typography>
      {scores.length === 0 ? (
        <Typography color="text.secondary">
          {polling ? "Grading in progress…" : "No criterion scores yet."}
        </Typography>
      ) : (
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Criterion</TableCell>
              <TableCell align="right">Score</TableCell>
              <TableCell align="right">Confidence</TableCell>
              <TableCell>Rationale</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {scores.map((row) => {
              const maxPts = 25;
              const frac = maxPts > 0 ? row.score / maxPts : 0;
              return (
                <TableRow key={row.criterion}>
                  <TableCell>{row.criterion}</TableCell>
                  <TableCell align="right">
                    <Chip
                      size="small"
                      label={row.score.toFixed(1)}
                      color={criterionChipColor(frac)}
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell align="right">{(row.confidence * 100).toFixed(0)}%</TableCell>
                  <TableCell sx={{ maxWidth: 480, whiteSpace: "pre-wrap" }}>{row.rationale}</TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      )}
    </Box>
  );
}
