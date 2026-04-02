import { useParams, Link as RouterLink } from "react-router-dom";
import { Box, Button, Card, CardContent, Typography } from "@mui/material";

export default function AssignmentDetail() {
  const { id } = useParams();

  return (
    <Card>
      <CardContent>
        <Typography variant="h3" component="h1" gutterBottom>
          Assignment #{id}
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Full assignment metadata will load here when the course-assignment API is connected.
        </Typography>
        <Button
          component={RouterLink}
          to={`/assignments/${id}/submit`}
          variant="contained"
          color="secondary"
          aria-label="Submit work for this assignment"
        >
          Submit work
        </Button>
      </CardContent>
    </Card>
  );
}
