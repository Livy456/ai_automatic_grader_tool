import { Box, Card, CardContent, Typography } from "@mui/material";
import GradeOutlined from "@mui/icons-material/GradeOutlined";

export default function MyGrades() {
  return (
    <Card>
      <CardContent sx={{ textAlign: "center", py: 6 }}>
        <GradeOutlined sx={{ fontSize: 56, color: "text.disabled", mb: 1 }} aria-hidden />
        <Typography variant="subtitle1" fontWeight={600}>
          No grades to show yet
        </Typography>
        <Typography variant="body2" color="text.secondary" maxWidth={400} sx={{ mx: "auto" }}>
          When your instructors publish scores for submitted work, they will appear here. Submit
          assignments from the Submit Assignment page to get started.
        </Typography>
      </CardContent>
    </Card>
  );
}
