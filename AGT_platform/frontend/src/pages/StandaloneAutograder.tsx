import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  LinearProgress,
  Step,
  StepLabel,
  Stepper,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Tabs,
  TextField,
  Typography,
} from "@mui/material";
import CloudUploadOutlined from "@mui/icons-material/CloudUploadOutlined";
import InsertDriveFileOutlined from "@mui/icons-material/InsertDriveFileOutlined";
import CloseOutlined from "@mui/icons-material/CloseOutlined";
import {
  listStandaloneSubmissions,
  submitStandalone,
  type StandaloneFileSpec,
  type StandaloneSubmissionSummary,
} from "../api";
import StatusChip from "../components/StatusChip";

const STEPS = ["Upload Files", "Add Context", "Review & Submit"];

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export default function StandaloneAutograder() {
  const navigate = useNavigate();
  const [tab, setTab] = useState(0);
  const [activeStep, setActiveStep] = useState(0);
  const [title, setTitle] = useState("");
  const [mainFiles, setMainFiles] = useState<File[]>([]);
  const [keyFile, setKeyFile] = useState<File | null>(null);
  const [rubricFile, setRubricFile] = useState<File | null>(null);
  const [keyText, setKeyText] = useState("");
  const [rubricText, setRubricText] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [history, setHistory] = useState<StandaloneSubmissionSummary[]>([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [historyLoading, setHistoryLoading] = useState(false);

  const loadHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const res = await listStandaloneSubmissions(1, 50);
      setHistory(res.items);
      setHistoryTotal(res.total);
    } catch {
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    if (tab === 1) void loadHistory();
  }, [tab, loadHistory]);

  const submit = async () => {
    if (!title.trim() || mainFiles.length === 0) return;
    setErr(null);
    setBusy(true);
    setProgress(0);

    const fileSpecs: StandaloneFileSpec[] = [];
    const filesToUpload: File[] = [];

    for (const f of mainFiles) {
      fileSpecs.push({
        filename: f.name,
        content_type: f.type || "application/octet-stream",
        artifact_kind: "submission",
      });
      filesToUpload.push(f);
    }
    if (rubricFile) {
      fileSpecs.push({
        filename: rubricFile.name,
        content_type: rubricFile.type || "application/octet-stream",
        artifact_kind: "rubric",
      });
      filesToUpload.push(rubricFile);
    }
    if (keyFile) {
      fileSpecs.push({
        filename: keyFile.name,
        content_type: keyFile.type || "application/octet-stream",
        artifact_kind: "answer_key",
      });
      filesToUpload.push(keyFile);
    }

    const n = filesToUpload.length || 1;
    try {
      const result = await submitStandalone(
        title.trim(),
        filesToUpload,
        fileSpecs,
        rubricText.trim() || undefined,
        keyText.trim() || undefined,
        (fileIndex, frac) => {
          setProgress(((fileIndex + frac) / n) * 100);
        },
      );
      setProgress(100);
      navigate(`/autograder/${result.submission_id}`);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const summaryFiles = [...mainFiles];
  if (rubricFile) summaryFiles.push(rubricFile);
  if (keyFile) summaryFiles.push(keyFile);

  const canNextStep0 = mainFiles.length > 0 && title.trim().length > 0;

  return (
    <Box>
      <Typography variant="h3" sx={{ mb: 2 }}>
        Standalone Autograder
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Grade files without enrolling in a course. Optional rubric and answer key improve accuracy.
      </Typography>

      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2 }} aria-label="Autograder sections">
        <Tab label="New Submission" id="autograder-tab-0" aria-controls="autograder-panel-0" />
        <Tab label="My Submissions" id="autograder-tab-1" aria-controls="autograder-panel-1" />
      </Tabs>

      {tab === 1 && (
        <Card>
          <CardContent>
            <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 2 }}>
              <Typography variant="h3">History</Typography>
              <Button size="small" onClick={() => void loadHistory()} disabled={historyLoading}>
                Refresh
              </Button>
            </Box>
            {historyLoading ? (
              <CircularProgress size={28} aria-label="Loading history" />
            ) : history.length === 0 ? (
              <Typography color="text.secondary">No standalone submissions yet.</Typography>
            ) : (
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Title</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell align="right">Score</TableCell>
                    <TableCell>Submitted</TableCell>
                    <TableCell align="right">Action</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {history.map((row) => (
                    <TableRow key={row.id}>
                      <TableCell>{row.title}</TableCell>
                      <TableCell>
                        <StatusChip status={row.status} />
                      </TableCell>
                      <TableCell align="right">
                        {row.final_score != null ? row.final_score.toFixed(1) : "—"}
                      </TableCell>
                      <TableCell>
                        {row.created_at
                          ? new Date(row.created_at).toLocaleString(undefined, {
                              dateStyle: "short",
                              timeStyle: "short",
                            })
                          : "—"}
                      </TableCell>
                      <TableCell align="right">
                        <Button size="small" onClick={() => navigate(`/autograder/${row.id}`)}>
                          View
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
            <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
              Total: {historyTotal}
            </Typography>
          </CardContent>
        </Card>
      )}

      {tab === 0 && (
        <>
          <Stepper activeStep={activeStep} sx={{ mb: 3 }} aria-label="Submission steps">
            {STEPS.map((label) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>

          {err && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {err}
            </Alert>
          )}

          {activeStep === 0 && (
            <Card>
              <CardContent>
                <TextField
                  label="Submission title"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  fullWidth
                  required
                  sx={{ mb: 2 }}
                  placeholder="e.g. My Essay Draft"
                  aria-label="Submission title"
                />
                <Box
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                  }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDragOver(false);
                    const list = e.dataTransfer.files;
                    if (list?.length) setMainFiles(Array.from(list));
                  }}
                  sx={{
                    border: "2px dashed",
                    borderColor: dragOver ? "secondary.main" : "divider",
                    borderRadius: 2,
                    p: 4,
                    textAlign: "center",
                    bgcolor: dragOver ? "action.hover" : "background.paper",
                  }}
                  aria-label="Drop zone for files to grade"
                >
                  <CloudUploadOutlined sx={{ fontSize: 48, color: "text.secondary", mb: 1 }} aria-hidden />
                  <Typography variant="subtitle1" gutterBottom>
                    Drag files to grade
                  </Typography>
                  <Button variant="outlined" component="label" sx={{ mt: 1 }} aria-label="Browse files">
                    Browse files
                    <input
                      type="file"
                      multiple
                      hidden
                      onChange={(e) => {
                        const f = e.target.files;
                        if (f?.length) setMainFiles(Array.from(f));
                      }}
                    />
                  </Button>
                  <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 2 }}>
                    PDF, DOCX, TXT, IPYNB, ZIP, MP4, PNG, JPG · Max 20 files (per server policy)
                  </Typography>
                </Box>
                {mainFiles.length > 0 && (
                  <Box sx={{ mt: 2 }}>
                    {mainFiles.map((f, i) => (
                      <Box
                        key={`${f.name}-${i}`}
                        sx={{
                          display: "flex",
                          alignItems: "center",
                          gap: 1,
                          py: 1,
                          borderBottom: 1,
                          borderColor: "divider",
                        }}
                      >
                        <InsertDriveFileOutlined color="action" aria-hidden />
                        <Box sx={{ flex: 1 }}>
                          <Typography variant="body2">{f.name}</Typography>
                          <Typography variant="caption" color="text.secondary">
                            {formatBytes(f.size)} · {f.type || "unknown type"}
                          </Typography>
                        </Box>
                        <Button
                          size="small"
                          aria-label={`Remove ${f.name}`}
                          onClick={() => setMainFiles((prev) => prev.filter((_, j) => j !== i))}
                        >
                          <CloseOutlined fontSize="small" />
                        </Button>
                      </Box>
                    ))}
                  </Box>
                )}
                <Box sx={{ display: "flex", justifyContent: "flex-end", mt: 2 }}>
                  <Button
                    variant="contained"
                    disabled={!canNextStep0}
                    onClick={() => setActiveStep(1)}
                    aria-label="Go to add context step"
                  >
                    Next
                  </Button>
                </Box>
              </CardContent>
            </Card>
          )}

          {activeStep === 1 && (
            <Card>
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>
                  Answer key / sample response (optional)
                </Typography>
                <Button variant="outlined" component="label" size="small" sx={{ mb: 1 }} aria-label="Upload answer key file">
                  Upload file
                  <input type="file" hidden onChange={(e) => setKeyFile(e.target.files?.[0] ?? null)} />
                </Button>
                {keyFile && (
                  <Typography variant="caption" display="block" color="text.secondary" sx={{ mb: 1 }}>
                    {keyFile.name}{" "}
                    <Button size="small" onClick={() => setKeyFile(null)} aria-label="Remove answer key file">
                      Remove
                    </Button>
                  </Typography>
                )}
                <TextField
                  fullWidth
                  multiline
                  minRows={3}
                  label="Or paste text"
                  value={keyText}
                  onChange={(e) => setKeyText(e.target.value)}
                  sx={{ mb: 3 }}
                  aria-label="Paste answer key or sample response"
                />

                <Typography variant="subtitle1" gutterBottom>
                  Rubric (optional)
                </Typography>
                <Button variant="outlined" component="label" size="small" sx={{ mb: 1 }} aria-label="Upload rubric file">
                  Upload file
                  <input type="file" hidden onChange={(e) => setRubricFile(e.target.files?.[0] ?? null)} />
                </Button>
                {rubricFile && (
                  <Typography variant="caption" display="block" color="text.secondary" sx={{ mb: 1 }}>
                    {rubricFile.name}{" "}
                    <Button size="small" onClick={() => setRubricFile(null)} aria-label="Remove rubric file">
                      Remove
                    </Button>
                  </Typography>
                )}
                <TextField
                  fullWidth
                  multiline
                  minRows={3}
                  label="Or paste rubric text"
                  value={rubricText}
                  onChange={(e) => setRubricText(e.target.value)}
                  aria-label="Paste rubric text"
                />

                <Box sx={{ display: "flex", justifyContent: "space-between", mt: 2 }}>
                  <Button onClick={() => setActiveStep(0)} aria-label="Back to upload step">
                    Back
                  </Button>
                  <Button variant="contained" onClick={() => setActiveStep(2)} aria-label="Go to review step">
                    Next
                  </Button>
                </Box>
              </CardContent>
            </Card>
          )}

          {activeStep === 2 && (
            <Card>
              <CardContent>
                <Typography variant="h3" sx={{ mb: 2 }}>
                  Review &amp; Submit
                </Typography>
                <Typography variant="body2">
                  <strong>Title:</strong> {title.trim() || "—"}
                </Typography>
                <Typography variant="body2" sx={{ mt: 1 }}>
                  Files ({summaryFiles.length}):{" "}
                  {summaryFiles.map((f) => f.name).join(", ") || "—"}
                </Typography>
                {(rubricText.trim() || keyText.trim()) && (
                  <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                    Includes pasted rubric and/or answer key text.
                  </Typography>
                )}
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                  Total size ~ {formatBytes(summaryFiles.reduce((s, f) => s + f.size, 0))}
                </Typography>
                {busy && (
                  <LinearProgress variant="determinate" value={progress} sx={{ mt: 2 }} aria-label="Upload progress" />
                )}
                <Button
                  variant="contained"
                  color="primary"
                  size="large"
                  fullWidth
                  sx={{ mt: 3 }}
                  disabled={busy || !canNextStep0}
                  onClick={submit}
                  aria-label="Submit for grading"
                >
                  {busy ? <CircularProgress size={22} color="inherit" aria-label="Submitting" /> : "Submit for Grading"}
                </Button>
                <Box sx={{ display: "flex", justifyContent: "flex-start", mt: 2 }}>
                  <Button onClick={() => setActiveStep(1)} disabled={busy} aria-label="Back to context step">
                    Back
                  </Button>
                </Box>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </Box>
  );
}
