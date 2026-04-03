import { Chip } from "@mui/material";
import type { ChipProps } from "@mui/material";

export type SubmissionStatus =
  | "queued"
  | "grading"
  | "graded"
  | "error"
  | "needs_review"
  | string;

type StatusChipProps = {
  status: SubmissionStatus;
  size?: ChipProps["size"];
};

const labelMap: Record<string, string> = {
  uploading: "Uploading",
  uploaded: "Uploaded",
  queued: "Queued",
  grading: "Grading",
  graded: "Graded",
  error: "Error",
  needs_review: "Needs review",
  deleted: "Deleted",
};

export default function StatusChip({ status, size = "small" }: StatusChipProps) {
  const normalized = String(status).toLowerCase();
  const label = labelMap[normalized] ?? status;

  let color: ChipProps["color"] = "default";
  if (normalized === "graded") color = "success";
  else if (normalized === "error") color = "error";
  else if (normalized === "grading" || normalized === "needs_review") color = "warning";

  return (
    <Chip
      size={size}
      label={label}
      color={color}
      variant={normalized === "queued" ? "outlined" : "filled"}
      aria-label={`Status: ${label}`}
    />
  );
}
