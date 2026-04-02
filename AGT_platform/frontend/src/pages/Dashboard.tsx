import { useEffect, useState } from "react";
import { Link, useLocation, Navigate } from "react-router-dom";
import { jwtDecode } from "jwt-decode";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  LinearProgress,
  Skeleton,
  Typography,
} from "@mui/material";
import InfoOutlined from "@mui/icons-material/InfoOutlined";
import SchoolOutlined from "@mui/icons-material/SchoolOutlined";
import { getToken } from "../auth";
import { listCourseAssignments, listCourses } from "../api";
import StatCard from "../components/StatCard";
import AdminDashboard from "./AdminDashboard";
import TeacherDashboard from "./TeacherDashboard";

interface JwtPayload {
  id: number;
  email: string;
  role: string;
}

type CourseCard = {
  id: number;
  code: string;
  title: string;
  instructor: string;
  assignmentCount: number;
  submitted: number;
};

// TODO: LTI 1.3 Integration
// Course enrollments should be fetched from Canvas via LTI Advantage (Names and Roles
// Provisioning Service). When the LTI tool is launched from Canvas, the platform passes
// a signed JWT (LTI launch token) containing the user's course context, role, and
// enrollment. The backend should exchange this for our internal JWT and sync enrollments.
//
// Reference: https://community.canvaslms.com/t5/Developers-Group/
//            LTI-1-3-and-LTI-Advantage-Implementation/td-p/255
//
// For now, fetch from GET /api/courses filtered by the user's enrollments.

const MOCK_COURSES: CourseCard[] = [
  {
    id: 1,
    code: "CS 101",
    title: "Introduction to Computer Science",
    instructor: "Dr. Example",
    assignmentCount: 4,
    submitted: 1,
  },
];

function useRole(): JwtPayload | null {
  const token = getToken();
  if (!token) return null;
  try {
    return jwtDecode<JwtPayload>(token);
  } catch {
    return null;
  }
}

function StudentDashboard() {
  const [courses, setCourses] = useState<CourseCard[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      try {
        const items = await listCourses();
        const courseList = Array.isArray(items) ? items : [];
        if (cancelled) return;
        if (courseList.length === 0) {
          setCourses(MOCK_COURSES);
        } else {
          const enriched = await Promise.all(
            courseList.map(async (c) => {
              let assignmentCount = 0;
              try {
                const assigns = await listCourseAssignments(c.id);
                assignmentCount = Array.isArray(assigns) ? assigns.length : 0;
              } catch {
                assignmentCount = 0;
              }
              return {
                id: c.id,
                code: c.code,
                title: c.title,
                instructor: "—",
                assignmentCount,
                submitted: 0,
              };
            }),
          );
          if (!cancelled) setCourses(enriched);
        }
      } catch {
        if (!cancelled) setCourses(MOCK_COURSES);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const enrolled = courses.length;
  const submitted = courses.reduce((s, c) => s + c.submitted, 0);
  const totalAssign = courses.reduce((s, c) => s + c.assignmentCount, 0);
  const avg =
    totalAssign > 0 ? Math.round((submitted / totalAssign) * 100) : "—";

  if (loading) {
    return (
      <Box aria-busy="true" aria-label="Loading dashboard">
        <Skeleton variant="rectangular" height={120} sx={{ mb: 2, borderRadius: 2 }} />
        <Skeleton variant="rectangular" height={200} sx={{ borderRadius: 2 }} />
      </Box>
    );
  }

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
        <StatCard title="Enrolled Courses" value={enrolled} aria-label="Enrolled courses count" />
        <StatCard title="Submitted Assignments" value={submitted} aria-label="Submitted assignments count" />
        <StatCard title="Avg Grade" value={avg === "—" ? "—" : `${avg}%`} aria-label="Average grade" />
      </Box>

      <Typography variant="h3" sx={{ mb: 2 }}>
        My Courses
      </Typography>

      <Alert severity="info" icon={<InfoOutlined />} sx={{ mb: 2 }}>
        Course enrollment is synced from Canvas via LTI 1.3 (Learning Tools Interoperability).
        When your institution connects Canvas to this platform, your enrolled courses and
        instructor assignments will appear here automatically. Contact your admin to enable
        the Canvas integration.
      </Alert>

      {courses.length === 0 ? (
        <Card>
          <CardContent sx={{ textAlign: "center", py: 6 }}>
            <SchoolOutlined sx={{ fontSize: 56, color: "text.disabled", mb: 1 }} aria-hidden />
            <Typography variant="subtitle1" fontWeight={600}>
              No courses yet
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Enrolled courses will show here after LTI sync or admin enrollment.
            </Typography>
          </CardContent>
        </Card>
      ) : (
        <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {courses.map((course) => {
            const pct =
              course.assignmentCount > 0
                ? Math.min(100, (course.submitted / course.assignmentCount) * 100)
                : 0;
            return (
              <Card key={course.id}>
                <CardContent>
                  <Box sx={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 1, mb: 1 }}>
                    <Chip label={course.code} size="small" color="secondary" variant="outlined" />
                    <Typography variant="h3" component="h2">
                      {course.title}
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Instructor: {course.instructor} · {course.assignmentCount} assignments
                  </Typography>
                  <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                    Progress: {course.submitted} of {course.assignmentCount} submitted
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={pct}
                    sx={{ height: 8, borderRadius: 1, mb: 2 }}
                    aria-label={`Progress ${course.submitted} of ${course.assignmentCount} assignments submitted`}
                  />
                  <Button
                    component={Link}
                    to="/assignments"
                    variant="contained"
                    color="secondary"
                    aria-label="View assignments for course"
                  >
                    View Assignments →
                  </Button>
                </CardContent>
              </Card>
            );
          })}
        </Box>
      )}
    </Box>
  );
}

export default function Dashboard() {
  const me = useRole();
  const location = useLocation();
  if (!me) return <Navigate to="/login" replace />;

  const adminPath = location.pathname === "/admin";

  if (adminPath) {
    if (me.role !== "admin") return <Navigate to="/" replace />;
    return <AdminDashboard />;
  }

  if (me.role === "admin") return <AdminDashboard />;
  if (me.role === "teacher") return <TeacherDashboard />;
  return <StudentDashboard />;
}
