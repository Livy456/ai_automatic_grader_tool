import { useState } from "react";
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  FormControl,
  IconButton,
  InputLabel,
  MenuItem,
  Select,
  TextField,
  Typography,
} from "@mui/material";
import DeleteOutline from "@mui/icons-material/DeleteOutline";
import AddOutlined from "@mui/icons-material/AddOutlined";
import { createCourseAssignment, type RubricCriterion } from "../api";

const MODALITIES = ["code", "written", "notebook", "video", "image"] as const;

export interface AssignmentFormProps {
  courseId: number;
  onSuccess: (assignment: { id: number; title: string }) => void;
  onCancel?: () => void;
}

type RubricRow = RubricCriterion;

export default function AssignmentForm({ courseId, onSuccess, onCancel }: AssignmentFormProps) {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [modality, setModality] = useState<string>("code");
  const [dueDate, setDueDate] = useState("");
  const [rubric, setRubric] = useState<RubricRow[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const addCriterion = () => {
    setRubric((r) => [...r, { criterion: "", max_score: 0 }]);
  };

  const updateCriterion = (i: number, patch: Partial<RubricRow>) => {
    setRubric((rows) => rows.map((row, j) => (j === i ? { ...row, ...patch } : row)));
  };

  const removeCriterion = (i: number) => {
    setRubric((rows) => rows.filter((_, j) => j !== i));
  };

  const submit = async () => {
    setErr(null);
    const t = title.trim();
    if (!t) {
      setErr("Title is required.");
      return;
    }
    if (t.length > 255) {
      setErr("Title must be at most 255 characters.");
      return;
    }
    if (!MODALITIES.includes(modality as (typeof MODALITIES)[number])) {
      setErr("Select a modality.");
      return;
    }
    const rubricPayload: RubricCriterion[] = rubric
      .filter((row) => row.criterion.trim())
      .map((row) => ({
        criterion: row.criterion.trim(),
        max_score: Number(row.max_score) || 0,
      }));

    setBusy(true);
    try {
      const res = await createCourseAssignment(courseId, {
        title: t,
        description: description.trim() || undefined,
        modality,
        rubric: rubricPayload.length ? rubricPayload : undefined,
        due_date: dueDate.trim() ? dueDate.trim() : null,
      });
      onSuccess({ id: res.id, title: res.title });
      setTitle("");
      setDescription("");
      setModality("code");
      setDueDate("");
      setRubric([]);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Box component="form" noValidate onSubmit={(e) => e.preventDefault()} aria-label="Create assignment">
      {err && (
        <Alert severity="error" sx={{ mb: 2 }} role="alert">
          {err}
        </Alert>
      )}
      <TextField
        label="Title"
        fullWidth
        required
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        inputProps={{ maxLength: 255 }}
        sx={{ mb: 2 }}
        aria-label="Assignment title"
      />
      <TextField
        label="Description"
        fullWidth
        multiline
        minRows={3}
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        sx={{ mb: 2 }}
        aria-label="Assignment description"
      />
      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel id={`modality-label-${courseId}`}>Modality</InputLabel>
        <Select
          labelId={`modality-label-${courseId}`}
          label="Modality"
          value={modality}
          onChange={(e) => setModality(e.target.value)}
          required
          aria-label="Assignment modality"
        >
          {MODALITIES.map((m) => (
            <MenuItem key={m} value={m}>
              {m}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      <TextField
        label="Due date"
        type="datetime-local"
        fullWidth
        value={dueDate}
        onChange={(e) => setDueDate(e.target.value)}
        InputLabelProps={{ shrink: true }}
        sx={{ mb: 2 }}
        aria-label="Due date"
      />
      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        Rubric (optional)
      </Typography>
      {rubric.map((row, i) => (
        <Box key={i} sx={{ display: "flex", gap: 1, mb: 1, alignItems: "flex-start" }}>
          <TextField
            label="Criterion"
            size="small"
            value={row.criterion}
            onChange={(e) => updateCriterion(i, { criterion: e.target.value })}
            sx={{ flex: 1 }}
            aria-label={`Rubric criterion ${i + 1}`}
          />
          <TextField
            label="Max score"
            type="number"
            size="small"
            value={row.max_score}
            onChange={(e) => updateCriterion(i, { max_score: Number(e.target.value) })}
            sx={{ width: 120 }}
            aria-label={`Max score for criterion ${i + 1}`}
          />
          <IconButton
            aria-label={`Remove criterion ${i + 1}`}
            onClick={() => removeCriterion(i)}
            color="error"
          >
            <DeleteOutline />
          </IconButton>
        </Box>
      ))}
      <Button startIcon={<AddOutlined />} onClick={addCriterion} sx={{ mb: 2 }} aria-label="Add rubric criterion">
        Add criterion
      </Button>
      <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
        <Button
          variant="contained"
          onClick={submit}
          disabled={busy}
          aria-label="Submit assignment"
        >
          {busy ? <CircularProgress size={20} color="inherit" aria-label="Saving" /> : "Create assignment"}
        </Button>
        {onCancel ? (
          <Button onClick={onCancel} aria-label="Cancel">
            Cancel
          </Button>
        ) : null}
      </Box>
    </Box>
  );
}
