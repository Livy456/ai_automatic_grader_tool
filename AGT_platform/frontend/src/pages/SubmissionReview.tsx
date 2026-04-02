import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { jwtDecode } from "jwt-decode";
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Collapse,
  IconButton,
  LinearProgress,
  Popover,
  Skeleton,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import ArrowBackOutlined from "@mui/icons-material/ArrowBackOutlined";
import ContentCopyOutlined from "@mui/icons-material/ContentCopyOutlined";
import ExpandMoreOutlined from "@mui/icons-material/ExpandMoreOutlined";
import { api } from "../api";
import { getToken } from "../auth";
import StatusChip from "../components/StatusChip";
import EvidenceBlock from "../components/EvidenceBlock";

const POLL_STATUSES = new Set(["uploading", "uploaded", "queued", "grading"]);

interface JwtPayload {
  id: number;
  email: string;
  role: string;
}

type AiScore = {
  criterion: string;
  score: number;
  confidence: number;
  rationale: string;
  evidence?: string[];
};

type SubmissionPayload = {
  id: number;
  status: string;
  final_score: number | null;
  final_feedback: string | null;
  grading_dispatch_at?: string | null;
  ai_scores?: AiScore[];
  rubric?: unknown;
  assignment_title?: string;
  student_name?: string;
  created_at?: string;
};

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

export default function SubmissionReview() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [sub, setSub] = useState<SubmissionPayload | null>(null);
  const [loadError, setLoadError] = useState(false);
  const [expanded, setExpanded] = useState<Record<number, boolean>>({});
  const [rubricOpen, setRubricOpen] = useState(false);
  const [evidenceAnchor, setEvidenceAnchor] = useState<{ el: HTMLElement; text: string } | null>(null);
  const [overrideScore, setOverrideScore] = useState("");
  const [overrideFeedback, setOverrideFeedback] = useState("");
  const [overrideBusy, setOverrideBusy] = useState(false);
  const [overrideErr, setOverrideErr] = useState<string | null>(null);

  const role = useMemo(() => {
    const t = getToken();
    if (!t) return "student";
    try {
      return jwtDecode<JwtPayload>(t).role;
    } catch {
      return "student";
    }
  }, []);

  const canOverride = role === "teacher" || role === "admin";

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | undefined;

    const tick = async () => {
      if (!id) return;
      try {
        const s = (await api.get(`/api/submissions/${id}`)) as SubmissionPayload;
        if (cancelled) return;
        setSub(s);
        setLoadError(false);
        setOverrideScore(s.final_score != null ? String(s.final_score) : "");
        setOverrideFeedback(s.final_feedback ?? "");
        const st = String(s.status);
        if (!POLL_STATUSES.has(st)) {
          if (timer) clearInterval(timer);
        }
      } catch {
        if (!cancelled) setLoadError(true);
      }
    };

    tick();
    timer = setInterval(tick, 2000);
    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [id]);

  const toggleCriterion = (idx: number) => {
    setExpanded((e) => ({ ...e, [idx]: !e[idx] }));
  };

  const copyJustification = useCallback(async () => {
    if (!sub) return;
    const parts: string[] = [];
    parts.push("Overall Feedback\n", sub.final_feedback ?? "", "\n\n");
    (sub.ai_scores ?? []).forEach((s, i) => {
      parts.push(`${i + 1}. ${s.criterion} (${s.score})\n`, s.rationale, "\n\n");
    });
    try {
      await navigator.clipboard.writeText(parts.join(""));
    } catch {
      /* ignore */
    }
  }, [sub]);

  const saveOverride = async () => {
    if (!id) return;
    setOverrideErr(null);
    setOverrideBusy(true);
    try {
      await api.post(`/api/teacher/submissions/${id}/override`, {
        final_score: Number(overrideScore),
        final_feedback: overrideFeedback,
      });
      const s = (await api.get(`/api/submissions/${id}`)) as SubmissionPayload;
      setSub(s);
    } catch (e: unknown) {
      setOverrideErr(e instanceof Error ? e.message : String(e));
    } finally {
      setOverrideBusy(false);
    }
  };

  if (!id) return null;

  if (!sub && !loadError) {
    return (
      <Box sx={{ py: 6, textAlign: "center" }} aria-busy="true" aria-label="Loading submission">
        <CircularProgress sx={{ mb: 2 }} />
        <Skeleton variant="rectangular" height={160} sx={{ borderRadius: 2, maxWidth: 800, mx: "auto" }} />
      </Box>
    );
  }

  if (loadError || !sub) {
    return (
      <Typography color="error" role="alert">
        Could not load submission.
      </Typography>
    );
  }

  const status = String(sub.status);
  const busy = POLL_STATUSES.has(status);
  const scoreNum = sub.final_score != null ? Number(sub.final_score) : null;
  const assignmentTitle = sub.assignment_title ?? `Submission #${sub.id}`;
  const submittedAt = sub.grading_dispatch_at ?? sub.created_at ?? "—";

  return (
    <Box>
      <Box sx={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 2, mb: 1 }}>
        <Button
          startIcon={<ArrowBackOutlined />}
          onClick={() => navigate(-1)}
          aria-label="Go back"
        >
          Back
        </Button>
        <Typography variant="h2" component="span" sx={{ flex: 1, minWidth: 200 }}>
          {assignmentTitle}
        </Typography>
        <StatusChip status={status} size="medium" />
      </Box>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Student: {sub.student_name ?? "—"} · Submitted: {submittedAt}
        {busy ? " · Status updates every few seconds while grading runs." : ""}
      </Typography>
      {busy && <LinearProgress sx={{ mb: 2 }} aria-label="Grading in progress" />}

      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", lg: "1fr 1fr" },
          gap: 2,
          alignItems: "start",
        }}
      >
        <Box>
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="overline" color="text.secondary">
                Total score
              </Typography>
              {scoreNum != null ? (
                <>
                  <Box sx={{ display: "flex", alignItems: "baseline", gap: 2, flexWrap: "wrap" }}>
                    <Typography variant="h2">
                      {scoreNum} / 100
                    </Typography>
                    <Typography variant="h3" color="text.secondary">
                      {Math.round(scoreNum)}%
                    </Typography>
                  </Box>
                  <LinearProgress
                    variant="determinate"
                    value={Math.min(100, scoreNum)}
                    color={gradeBarColor(scoreNum)}
                    sx={{ height: 10, borderRadius: 1, mt: 2 }}
                    aria-label={`Score ${scoreNum} percent`}
                  />
                </>
              ) : (
                <Typography color="text.secondary">No final score yet</Typography>
              )}
            </CardContent>
          </Card>

          <Typography variant="subtitle2" color="text.secondary" sx={{ mb: 1 }}>
            Rubric scorecard
          </Typography>
          {(sub.ai_scores ?? []).map((s, idx) => {
            const maxGuess = s.score <= 10 ? 10 : 100;
            const fraction = s.score / maxGuess;
            return (
              <Card key={idx} sx={{ mb: 1.5 }}>
                <CardContent sx={{ py: 2 }}>
                  <Box
                    sx={{
                      display: "flex",
                      flexWrap: "wrap",
                      alignItems: "center",
                      gap: 1,
                      cursor: "pointer",
                    }}
                    onClick={() => toggleCriterion(idx)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        toggleCriterion(idx);
                      }
                    }}
                    role="button"
                    tabIndex={0}
                    aria-expanded={Boolean(expanded[idx])}
                    aria-label={`Criterion ${s.criterion}, toggle rationale`}
                  >
                    <Typography variant="subtitle1" sx={{ flex: 1 }}>
                      {s.criterion}
                    </Typography>
                    <Chip
                      size="small"
                      label={`${s.score} / ${maxGuess}`}
                      color={criterionChipColor(fraction)}
                      variant="outlined"
                    />
                    <IconButton size="small" aria-hidden tabIndex={-1}>
                      <ExpandMoreOutlined
                        sx={{
                          transform: expanded[idx] ? "rotate(180deg)" : "none",
                          transition: "transform 0.2s",
                        }}
                      />
                    </IconButton>
                  </Box>
                  <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                    Confidence: {Math.round((s.confidence <= 1 ? s.confidence * 100 : s.confidence))}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={Math.min(100, s.confidence <= 1 ? s.confidence * 100 : s.confidence)}
                    sx={{ height: 4, borderRadius: 1, mt: 0.5 }}
                    aria-label={`Confidence ${s.confidence}`}
                  />
                  <Collapse in={Boolean(expanded[idx])}>
                    <Typography variant="body2" sx={{ mt: 2, whiteSpace: "pre-wrap" }}>
                      {s.rationale}
                    </Typography>
                  </Collapse>
                </CardContent>
              </Card>
            );
          })}
          {(sub.ai_scores ?? []).length === 0 && (
            <Typography variant="body2" color="text.secondary">
              No per-criterion scores yet.
            </Typography>
          )}
        </Box>

        <Card>
          <CardContent>
            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2 }}>
              <Typography variant="overline" color="text.secondary">
                Grader justification
              </Typography>
              <Button
                size="small"
                startIcon={<ContentCopyOutlined />}
                onClick={copyJustification}
                aria-label="Copy justification to clipboard"
              >
                Copy
              </Button>
            </Box>

            <Typography variant="subtitle2" gutterBottom>
              Overall feedback
            </Typography>
            <Typography variant="body2" sx={{ whiteSpace: "pre-wrap", mb: 3 }}>
              {sub.final_feedback ?? "—"}
            </Typography>

            <Typography variant="subtitle2" gutterBottom>
              Per-criterion breakdown
            </Typography>
            {(sub.ai_scores ?? []).map((s, idx) => (
              <Box key={idx} sx={{ mb: 2 }}>
                <Typography variant="subtitle2" component="h3">
                  ▸ {s.criterion}{" "}
                  <Chip size="small" label={String(s.score)} sx={{ ml: 0.5 }} />
                </Typography>
                <Typography variant="body2" sx={{ mt: 0.5, whiteSpace: "pre-wrap" }}>
                  {s.rationale}
                </Typography>
                {(s.evidence ?? []).map((ev, j) => (
                  <EvidenceBlock key={j} text={ev} index={j} />
                ))}
                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mt: 1 }}>
                  {(s.evidence ?? []).map((ev, j) => (
                    <Button
                      key={j}
                      size="small"
                      variant="outlined"
                      onClick={(e) => setEvidenceAnchor({ el: e.currentTarget, text: ev })}
                      aria-label={`Evidence ${j + 1}, open full text`}
                    >
                      Evidence {j + 1}
                    </Button>
                  ))}
                </Box>
              </Box>
            ))}

            <Box sx={{ mt: 3 }}>
              <Button
                size="small"
                onClick={() => setRubricOpen((o) => !o)}
                aria-expanded={rubricOpen}
                aria-label="Toggle rubric reference"
              >
                Rubric reference {rubricOpen ? "▾" : "▸"}
              </Button>
              <Collapse in={rubricOpen}>
                <Box sx={{ mt: 2, overflow: "auto" }}>
                  {sub.rubric != null && typeof sub.rubric === "object" && !Array.isArray(sub.rubric) ? (
                    <Table size="small" aria-label="Rubric reference table">
                      <TableHead>
                        <TableRow>
                          <TableCell>Key</TableCell>
                          <TableCell>Value</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {Object.entries(sub.rubric as Record<string, unknown>).map(([k, v]) => (
                          <TableRow key={k}>
                            <TableCell>{k}</TableCell>
                            <TableCell>{typeof v === "object" ? JSON.stringify(v) : String(v)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  ) : Array.isArray(sub.rubric) ? (
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>#</TableCell>
                          <TableCell>Entry</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {(sub.rubric as unknown[]).map((row, i) => (
                          <TableRow key={i}>
                            <TableCell>{i + 1}</TableCell>
                            <TableCell>{JSON.stringify(row)}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      No rubric payload on this submission (API may omit it).
                    </Typography>
                  )}
                </Box>
              </Collapse>
            </Box>
          </CardContent>
        </Card>
      </Box>

      {canOverride && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="h3" gutterBottom>
              Override (teacher)
            </Typography>
            <TextField
              label="Final score"
              type="number"
              fullWidth
              value={overrideScore}
              onChange={(e) => setOverrideScore(e.target.value)}
              sx={{ mb: 2 }}
              aria-label="Override final score"
            />
            <TextField
              label="Final feedback"
              fullWidth
              multiline
              minRows={4}
              value={overrideFeedback}
              onChange={(e) => setOverrideFeedback(e.target.value)}
              sx={{ mb: 2 }}
              aria-label="Override final feedback"
            />
            {overrideErr && (
              <Typography color="error" variant="body2" sx={{ mb: 1 }} role="alert">
                {overrideErr}
              </Typography>
            )}
            <Button
              variant="contained"
              onClick={saveOverride}
              disabled={overrideBusy}
              aria-label="Save grade override"
            >
              {overrideBusy ? <CircularProgress size={18} color="inherit" aria-label="Saving" /> : "Save Override"}
            </Button>
          </CardContent>
        </Card>
      )}

      <Popover
        open={Boolean(evidenceAnchor)}
        anchorEl={evidenceAnchor?.el}
        onClose={() => setEvidenceAnchor(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
      >
        <Box sx={{ p: 2, maxWidth: 360 }}>
          <Typography variant="body2" sx={{ whiteSpace: "pre-wrap" }}>
            {evidenceAnchor?.text}
          </Typography>
        </Box>
      </Popover>
    </Box>
  );
}
