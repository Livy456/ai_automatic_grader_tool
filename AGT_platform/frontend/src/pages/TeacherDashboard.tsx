import { useCallback, useEffect, useState } from "react";
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Dialog,
  DialogContent,
  DialogTitle,
  Typography,
} from "@mui/material";
import AddOutlined from "@mui/icons-material/AddOutlined";
import { listCourseAssignments, listCourses, type CourseListItem } from "../api";
import AssignmentForm from "../components/AssignmentForm";
import StatCard from "../components/StatCard";
import TeacherSubmissionsGrid from "../components/TeacherSubmissionsGrid";

type TeacherCourse = CourseListItem & { assignmentCount: number };

export default function TeacherDashboard() {
  const [courses, setCourses] = useState<TeacherCourse[]>([]);
  const [dialogCourseId, setDialogCourseId] = useState<number | null>(null);

  const loadCourses = useCallback(async () => {
    try {
      const all = await listCourses();
      const teacherRows = (Array.isArray(all) ? all : []).filter(
        (c) => c.enrollment_role === "teacher",
      );
      const enriched = await Promise.all(
        teacherRows.map(async (c) => {
          try {
            const assigns = await listCourseAssignments(c.id);
            return {
              ...c,
              assignmentCount: Array.isArray(assigns) ? assigns.length : 0,
            };
          } catch {
            return { ...c, assignmentCount: 0 };
          }
        }),
      );
      setCourses(enriched);
    } catch {
      setCourses([]);
    }
  }, []);

  useEffect(() => {
    void loadCourses();
  }, [loadCourses]);

  const dialogCourse = courses.find((c) => c.id === dialogCourseId);

  return (
    <Box>
      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", sm: "repeat(3, 1fr)" },
          gap: 2,
          mb: 3,
        }}
      >
        <StatCard title="Courses" value="—" subtitle="LTI sync pending" aria-label="Courses count" />
        <StatCard title="Total Submissions" value="—" aria-label="Total submissions" />
        <StatCard title="Pending Review" value="—" aria-label="Pending review count" />
      </Box>

      <Typography variant="h3" sx={{ mb: 2 }}>
        My Courses
      </Typography>
      {courses.length === 0 ? (
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          No courses where you are listed as a teacher. Ask an admin to enroll you as a teacher on a course.
        </Typography>
      ) : (
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: { xs: "1fr", md: "repeat(2, 1fr)" },
            gap: 2,
            mb: 3,
          }}
        >
          {courses.map((course) => (
            <Card key={course.id}>
              <CardContent>
                <Box sx={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 1, mb: 1 }}>
                  <Chip label={course.code} size="small" color="secondary" variant="outlined" />
                  <Typography variant="h3" component="h2">
                    {course.title}
                  </Typography>
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  You are a teacher · {course.assignmentCount} assignment
                  {course.assignmentCount === 1 ? "" : "s"}
                </Typography>
                <Button
                  variant="contained"
                  color="secondary"
                  startIcon={<AddOutlined />}
                  onClick={() => setDialogCourseId(course.id)}
                  aria-label={`New assignment for ${course.code}`}
                >
                  New Assignment
                </Button>
              </CardContent>
            </Card>
          ))}
        </Box>
      )}

      <Typography variant="h3" sx={{ mb: 2 }}>
        Recent Submissions
      </Typography>
      <TeacherSubmissionsGrid />

      <Dialog
        open={dialogCourseId != null}
        onClose={() => setDialogCourseId(null)}
        fullWidth
        maxWidth="sm"
        aria-labelledby="teacher-new-assignment-title"
      >
        <DialogTitle id="teacher-new-assignment-title">
          New assignment — {dialogCourse?.code ?? ""}
        </DialogTitle>
        <DialogContent>
          {dialogCourseId != null && (
            <AssignmentForm
              courseId={dialogCourseId}
              onCancel={() => setDialogCourseId(null)}
              onSuccess={async () => {
                setDialogCourseId(null);
                await loadCourses();
              }}
            />
          )}
        </DialogContent>
      </Dialog>
    </Box>
  );
}
