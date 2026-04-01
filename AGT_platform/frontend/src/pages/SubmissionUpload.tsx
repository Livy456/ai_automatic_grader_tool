import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { submitAssignmentDirect } from "../api";
import { Container, Typography, Button, LinearProgress, Alert } from "@mui/material";

/**
 * Student uploads go: API presign → browser PUT to S3 → finalize API.
 * Files never stream through Flask in production (scalability + memory).
 */
export default function SubmissionUpload() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [files, setFiles] = useState<FileList | null>(null);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [err, setErr] = useState<string | null>(null);

  const submit = async () => {
    if (!files || !id) return;
    setErr(null);
    setBusy(true);
    setProgress(0);
    const list = Array.from(files);
    const n = list.length || 1;
    try {
      const aid = parseInt(id, 10);
      const result = await submitAssignmentDirect(aid, list, (fileIndex, frac) => {
        setProgress(((fileIndex + frac) / n) * 100);
      });
      setProgress(100);
      navigate(`/submissions/${result.submission_id}`);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Container sx={{ mt: 3 }}>
      <Typography variant="h5" gutterBottom>
        Upload Submission
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Files upload directly to object storage. The API only issues short-lived upload URLs and
        starts grading after upload completes.
      </Typography>
      {err && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {err}
        </Alert>
      )}
      <input
        type="file"
        multiple
        disabled={busy}
        onChange={(e) => setFiles(e.target.files)}
      />
      <div style={{ marginTop: 16 }}>
        <Button variant="contained" onClick={submit} disabled={busy || !files?.length}>
          {busy ? "Uploading…" : "Submit"}
        </Button>
      </div>
      {busy && <LinearProgress variant="determinate" value={progress} sx={{ mt: 2 }} />}
    </Container>
  );
}
