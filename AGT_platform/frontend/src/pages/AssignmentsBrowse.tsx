import { Box, Button, Card, CardContent, Chip, Typography } from "@mui/material";
import AssignmentOutlined from "@mui/icons-material/AssignmentOutlined";
import { Link } from "react-router-dom";

/**
 * Course-assignment list API is not wired to this route yet.
 * Placeholder rows let students reach /assignments/:id/submit for demos.
 */
const PLACEHOLDER_ASSIGNMENTS = [
  { id: 1, title: "Homework 1 — Foundations", modality: "written" },
  { id: 2, title: "Project milestone", modality: "mixed" },
];

export default function AssignmentsBrowse() {
  return (
    <Box>
      <Typography variant="h3" sx={{ mb: 2 }}>
        Assignments
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Open an assignment to upload your submission. Your school may sync these from Canvas via LTI.
      </Typography>
      {PLACEHOLDER_ASSIGNMENTS.length === 0 ? (
        <Card>
          <CardContent sx={{ textAlign: "center", py: 6 }}>
            <AssignmentOutlined sx={{ fontSize: 56, color: "text.disabled", mb: 1 }} aria-hidden />
            <Typography variant="subtitle1" fontWeight={600}>
              No assignments
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Check back after your instructor publishes assignments.
            </Typography>
          </CardContent>
        </Card>
      ) : (
        <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {PLACEHOLDER_ASSIGNMENTS.map((a) => (
            <Card key={a.id}>
              <CardContent sx={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 2 }}>
                <Box sx={{ flex: 1, minWidth: 200 }}>
                  <Typography variant="h3" component="h2">
                    {a.title}
                  </Typography>
                  <Chip label={a.modality} size="small" sx={{ mt: 1 }} aria-label={`Modality ${a.modality}`} />
                </Box>
                <Button
                  component={Link}
                  to={`/assignments/${a.id}/submit`}
                  variant="contained"
                  color="secondary"
                  aria-label={`Submit assignment ${a.title}`}
                >
                  Submit
                </Button>
                <Button component={Link} to={`/assignments/${a.id}`} variant="outlined" aria-label={`View assignment ${a.title}`}>
                  Details
                </Button>
              </CardContent>
            </Card>
          ))}
        </Box>
      )}
    </Box>
  );
}
