import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemText,
  MenuItem,
  Snackbar,
  Tab,
  Tabs,
  TextField,
  Typography,
} from "@mui/material";
import AssignmentOutlined from "@mui/icons-material/AssignmentOutlined";
import PeopleOutlined from "@mui/icons-material/PeopleOutlined";
import DeleteOutline from "@mui/icons-material/DeleteOutline";
import { DataGrid, type GridColDef, type GridRenderCellParams } from "@mui/x-data-grid";
import {
  adminEnrollUser,
  adminListCourseEnrollments,
  adminRemoveEnrollment,
  api,
  listCourseAssignments,
  type CourseAssignment,
  type CourseDetail,
} from "../api";
import AssignmentForm from "../components/AssignmentForm";
import StatCard from "../components/StatCard";

type AdminUser = { id: number; email: string; name?: string; role: string };
type AdminCourseRow = {
  id: number;
  code: string;
  title: string;
  description?: string | null;
  enrollment_count: number;
};
type AuditRow = {
  id: string;
  time: string;
  actor_user_id: number | null;
  action: string;
  target_type: string;
  target_id: number;
};

// Module-level cache — survives unmount/remount from React Router. Populated after the
// first successful refresh(); remounts show cached data immediately while a background
// refresh runs (stale-while-revalidate).
type AdminCache = {
  users: AdminUser[];
  courses: AdminCourseRow[];
  audit: AuditRow[];
  roleDraft: Record<number, string>;
};
let _cache: AdminCache | null = null;

function AdminUsersEmptyOverlay() {
  return (
    <Box sx={{ py: 6, textAlign: "center", color: "text.secondary" }}>
      <Typography>No users loaded</Typography>
    </Box>
  );
}

export default function AdminDashboard() {
  const [tab, setTab] = useState(0);
  const [users, setUsers] = useState<AdminUser[]>(_cache?.users ?? []);
  const [courses, setCourses] = useState<AdminCourseRow[]>(_cache?.courses ?? []);
  const [audit, setAudit] = useState<AuditRow[]>(_cache?.audit ?? []);
  const [loading, setLoading] = useState(_cache === null);
  const [newCode, setNewCode] = useState("");
  const [newTitle, setNewTitle] = useState("");
  const [newDescription, setNewDescription] = useState("");
  const [creating, setCreating] = useState(false);
  const [roleDraft, setRoleDraft] = useState<Record<number, string>>(_cache?.roleDraft ?? {});
  const [savingId, setSavingId] = useState<number | null>(null);
  const [snackbar, setSnackbar] = useState<{ open: boolean; message: string; severity: "success" | "error" }>({
    open: false,
    message: "",
    severity: "success",
  });

  const [enrollmentCourseId, setEnrollmentCourseId] = useState<number | null>(null);
  const [enrollments, setEnrollments] = useState<CourseDetail["enrollments"]>([]);
  const [enrollUserId, setEnrollUserId] = useState<number | "">("");
  const [enrollRole, setEnrollRole] = useState<"student" | "teacher">("student");
  const [enrollBusy, setEnrollBusy] = useState(false);
  const [enrollWarning, setEnrollWarning] = useState<string | null>(null);

  const [assignmentCourseId, setAssignmentCourseId] = useState<number | null>(null);
  const [assignments, setAssignments] = useState<CourseAssignment[]>([]);

  const refresh = useCallback(async () => {
    if (_cache === null) {
      setLoading(true);
    }
    try {
      const [u, c, a] = await Promise.all([
        api.get("/api/admin/users") as Promise<AdminUser[]>,
        api.get("/api/admin/courses") as Promise<AdminCourseRow[]>,
        api.get("/api/admin/audit") as Promise<
          Array<{ time: string; actor_user_id: number | null; action: string; target_type: string; target_id: number }>
        >,
      ]);

      const nextUsers = Array.isArray(u) ? u : [];
      const courseRows = Array.isArray(c) ? c : [];
      const nextCourses = courseRows.map((row) => ({
        ...row,
        enrollment_count: typeof row.enrollment_count === "number" ? row.enrollment_count : 0,
      }));
      const nextAudit = (Array.isArray(a) ? a : []).map((row, i) => ({
        ...row,
        id: `audit-${row.time}-${i}`,
      }));
      const rd: Record<number, string> = {};
      nextUsers.forEach((x) => {
        rd[x.id] = x.role;
      });

      _cache = { users: nextUsers, courses: nextCourses, audit: nextAudit, roleDraft: rd };

      setUsers(nextUsers);
      setCourses(nextCourses);
      setAudit(nextAudit);
      setRoleDraft(rd);
    } catch (err) {
      console.error("[AdminDashboard] refresh failed:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const loadEnrollments = useCallback(async (courseId: number) => {
    try {
      const rows = await adminListCourseEnrollments(courseId);
      setEnrollments(Array.isArray(rows) ? rows : []);
    } catch {
      setEnrollments([]);
    }
  }, []);

  const loadAssignments = useCallback(async (courseId: number) => {
    try {
      const rows = await listCourseAssignments(courseId);
      setAssignments(Array.isArray(rows) ? rows : []);
    } catch {
      setAssignments([]);
    }
  }, []);

  useEffect(() => {
    if (enrollmentCourseId != null) {
      void loadEnrollments(enrollmentCourseId);
      setEnrollWarning(null);
      setEnrollUserId("");
      setEnrollRole("student");
    }
  }, [enrollmentCourseId, loadEnrollments]);

  useEffect(() => {
    if (assignmentCourseId != null) {
      void loadAssignments(assignmentCourseId);
    }
  }, [assignmentCourseId, loadAssignments]);

  const todayStart = useMemo(() => {
    const d = new Date();
    d.setHours(0, 0, 0, 0);
    return d.getTime();
  }, []);

  const submissionsToday = audit.filter(
    (r) =>
      r.action === "CREATE_SUBMISSION" &&
      new Date(r.time).getTime() >= todayStart,
  ).length;

  const errorsToday = audit.filter(
    (r) =>
      String(r.action).toLowerCase().includes("error") &&
      new Date(r.time).getTime() >= todayStart,
  ).length;

  const saveRole = async (userId: number) => {
    const role = roleDraft[userId];
    if (!role) return;
    setSavingId(userId);
    try {
      await api.post(`/api/admin/users/${userId}/role`, { role });
      await refresh();
    } finally {
      setSavingId(null);
    }
  };

  const createCourse = async () => {
    if (!newCode.trim() || !newTitle.trim()) return;
    setCreating(true);
    try {
      await api.post("/api/admin/courses", {
        code: newCode.trim(),
        title: newTitle.trim(),
        description: newDescription.trim() || undefined,
      });
      setNewCode("");
      setNewTitle("");
      setNewDescription("");
      setSnackbar({ open: true, message: "Course created", severity: "success" });
      await refresh();
    } catch (e: unknown) {
      setSnackbar({
        open: true,
        message: e instanceof Error ? e.message : "Failed to create course",
        severity: "error",
      });
    } finally {
      setCreating(false);
    }
  };

  const removeEnrollment = async (enrollmentId: number) => {
    try {
      await adminRemoveEnrollment(enrollmentId);
      if (enrollmentCourseId != null) await loadEnrollments(enrollmentCourseId);
      await refresh();
    } catch (e: unknown) {
      setSnackbar({
        open: true,
        message: e instanceof Error ? e.message : "Remove failed",
        severity: "error",
      });
    }
  };

  const submitEnroll = async () => {
    if (enrollmentCourseId == null || enrollUserId === "") return;
    setEnrollBusy(true);
    setEnrollWarning(null);
    try {
      await adminEnrollUser({
        course_id: enrollmentCourseId,
        user_id: Number(enrollUserId),
        role: enrollRole,
      });
      await loadEnrollments(enrollmentCourseId);
      await refresh();
      setEnrollUserId("");
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      if (msg.includes("409")) {
        setEnrollWarning("This user is already enrolled in this course.");
      } else {
        setSnackbar({ open: true, message: msg, severity: "error" });
      }
    } finally {
      setEnrollBusy(false);
    }
  };

  const userColumns: GridColDef<AdminUser>[] = [
    { field: "id", headerName: "ID", width: 70 },
    { field: "email", headerName: "Email", flex: 1, minWidth: 180 },
    {
      field: "name",
      headerName: "Name",
      flex: 0.8,
      minWidth: 120,
      valueGetter: (_v: unknown, row: AdminUser) => row.name ?? "—",
    },
    {
      field: "role",
      headerName: "Role",
      width: 160,
      renderCell: (params: GridRenderCellParams<AdminUser>) => (
        <TextField
          select
          size="small"
          value={roleDraft[params.row.id] ?? params.row.role}
          onChange={(e) =>
            setRoleDraft((d) => ({ ...d, [params.row.id]: e.target.value }))
          }
          aria-label={`Role for user ${params.row.email}`}
        >
          <MenuItem value="student">student</MenuItem>
          <MenuItem value="teacher">teacher</MenuItem>
          <MenuItem value="admin">admin</MenuItem>
        </TextField>
      ),
    },
    {
      field: "last_login",
      headerName: "Last Login",
      width: 120,
      valueGetter: () => "—",
    },
    {
      field: "actions",
      headerName: "Actions",
      width: 110,
      sortable: false,
      renderCell: (params: GridRenderCellParams<AdminUser>) => (
        <Button
          size="small"
          variant="contained"
          disabled={savingId === params.row.id}
          onClick={() => saveRole(params.row.id)}
          aria-label={`Save role for user ${params.row.id}`}
        >
          Save
        </Button>
      ),
    },
  ];

  const courseColumns: GridColDef<AdminCourseRow>[] = [
    { field: "id", headerName: "ID", width: 80 },
    { field: "code", headerName: "Code", flex: 0.5, minWidth: 100 },
    { field: "title", headerName: "Title", flex: 1, minWidth: 160 },
    { field: "enrollment_count", headerName: "# Enrolled", width: 110, type: "number" },
    {
      field: "actions",
      headerName: "Actions",
      width: 120,
      sortable: false,
      renderCell: (params: GridRenderCellParams<AdminCourseRow>) => (
        <Box>
          <IconButton
            size="small"
            aria-label="Open enrollments for course"
            onClick={() => setEnrollmentCourseId(params.row.id)}
          >
            <PeopleOutlined fontSize="small" />
          </IconButton>
          <IconButton
            size="small"
            aria-label="Open assignments for course"
            onClick={() => setAssignmentCourseId(params.row.id)}
          >
            <AssignmentOutlined fontSize="small" />
          </IconButton>
        </Box>
      ),
    },
  ];

  const auditColumns: GridColDef<AuditRow>[] = [
    { field: "time", headerName: "Time", flex: 1, minWidth: 180 },
    { field: "actor_user_id", headerName: "Actor", width: 90 },
    { field: "action", headerName: "Action", flex: 0.8, minWidth: 140 },
    {
      field: "target",
      headerName: "Target",
      flex: 1,
      minWidth: 160,
      valueGetter: (_v: unknown, row: AuditRow) => `${row.target_type} #${row.target_id}`,
    },
  ];

  const enrollmentCourse = courses.find((c) => c.id === enrollmentCourseId);
  const assignmentCourse = courses.find((c) => c.id === assignmentCourseId);

  if (loading && users.length === 0 && courses.length === 0) {
    return (
      <Box aria-busy="true">
        <Typography variant="body2" color="text.secondary">
          Loading admin data…
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert
          onClose={() => setSnackbar((s) => ({ ...s, open: false }))}
          severity={snackbar.severity}
          variant="filled"
          sx={{ width: "100%" }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>

      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", md: "repeat(4, 1fr)" },
          gap: 2,
          mb: 3,
        }}
      >
        <StatCard title="Users" value={users.length} aria-label="User count" />
        <StatCard title="Courses" value={courses.length} aria-label="Course count" />
        <StatCard title="Submissions Today" value={submissionsToday} aria-label="Submissions created today" />
        <StatCard title="Errors" value={errorsToday} aria-label="Error events today" />
      </Box>

      <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2 }} aria-label="Admin sections">
        <Tab label="Users" id="admin-tab-users" aria-controls="admin-panel-users" />
        <Tab label="Courses" id="admin-tab-courses" aria-controls="admin-panel-courses" />
        <Tab label="Audit Log" id="admin-tab-audit" aria-controls="admin-panel-audit" />
      </Tabs>

      {tab === 0 && (
        <Box id="admin-panel-users" role="tabpanel" aria-labelledby="admin-tab-users">
          <DataGrid
            rows={users}
            columns={userColumns}
            autoHeight
            pageSizeOptions={[25]}
            initialState={{ pagination: { paginationModel: { pageSize: 25 } } }}
            slots={{ noRowsOverlay: AdminUsersEmptyOverlay }}
            sx={{ border: 1, borderColor: "divider", borderRadius: 2 }}
            aria-label="Users"
          />
        </Box>
      )}

      {tab === 1 && (
        <Box id="admin-panel-courses" role="tabpanel" aria-labelledby="admin-tab-courses">
          <Typography variant="h3" sx={{ mb: 2 }}>
            Create course
          </Typography>
          <Box sx={{ display: "flex", flexDirection: "column", gap: 2, maxWidth: 520, mb: 3 }}>
            <TextField
              label="Code"
              value={newCode}
              onChange={(e) => setNewCode(e.target.value)}
              aria-label="Course code"
            />
            <TextField
              label="Title"
              value={newTitle}
              onChange={(e) => setNewTitle(e.target.value)}
              aria-label="Course title"
            />
            <TextField
              label="Description"
              value={newDescription}
              onChange={(e) => setNewDescription(e.target.value)}
              multiline
              minRows={3}
              aria-label="Course description"
            />
            <Button
              variant="contained"
              onClick={createCourse}
              disabled={creating || !newCode.trim() || !newTitle.trim()}
              aria-label="Create course"
            >
              {creating ? "Creating…" : "Create"}
            </Button>
          </Box>

          <Typography variant="h3" sx={{ mb: 2 }}>
            All courses
          </Typography>
          <DataGrid
            rows={courses}
            columns={courseColumns}
            autoHeight
            pageSizeOptions={[25]}
            initialState={{ pagination: { paginationModel: { pageSize: 25 } } }}
            slots={{
              noRowsOverlay: () => (
                <Box sx={{ py: 4, textAlign: "center", color: "text.secondary" }}>No courses yet</Box>
              ),
            }}
            sx={{ border: 1, borderColor: "divider", borderRadius: 2 }}
            aria-label="Courses"
          />

          <Drawer anchor="right" open={enrollmentCourseId != null} onClose={() => setEnrollmentCourseId(null)}>
            <Box sx={{ width: 400, p: 3 }} role="region" aria-label="Course enrollments">
              <Typography variant="h3" sx={{ mb: 2 }}>
                Enrollments — {enrollmentCourse?.code ?? ""}
              </Typography>
              {enrollWarning && (
                <Alert severity="warning" sx={{ mb: 2 }} onClose={() => setEnrollWarning(null)}>
                  {enrollWarning}
                </Alert>
              )}
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Current enrollments
              </Typography>
              <List dense sx={{ mb: 2, border: 1, borderColor: "divider", borderRadius: 1 }}>
                {enrollments.length === 0 ? (
                  <ListItem>
                    <ListItemText primary="No enrollments yet" />
                  </ListItem>
                ) : (
                  enrollments.map((row) => (
                    <ListItem
                      key={row.enrollment_id ?? `${row.user_id}-${row.role}`}
                      secondaryAction={
                        row.enrollment_id != null ? (
                          <IconButton
                            edge="end"
                            aria-label={`Remove enrollment for ${row.email}`}
                            onClick={() => removeEnrollment(row.enrollment_id!)}
                          >
                            <DeleteOutline />
                          </IconButton>
                        ) : null
                      }
                    >
                      <ListItemText
                        primary={`${row.name || row.email} (${row.email})`}
                        secondary={`Role: ${row.role}`}
                      />
                    </ListItem>
                  ))
                )}
              </List>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Add enrollment
              </Typography>
              <TextField
                select
                fullWidth
                label="User"
                value={enrollUserId}
                onChange={(e) => setEnrollUserId(e.target.value === "" ? "" : Number(e.target.value))}
                sx={{ mb: 2 }}
                aria-label="Select user to enroll"
              >
                <MenuItem value="">
                  <em>Select user</em>
                </MenuItem>
                {users.map((u) => (
                  <MenuItem key={u.id} value={u.id}>
                    {u.name || u.email} ({u.email})
                  </MenuItem>
                ))}
              </TextField>
              <TextField
                select
                fullWidth
                label="Role"
                value={enrollRole}
                onChange={(e) => setEnrollRole(e.target.value as "student" | "teacher")}
                sx={{ mb: 2 }}
                aria-label="Enrollment role"
              >
                <MenuItem value="student">student</MenuItem>
                <MenuItem value="teacher">teacher</MenuItem>
              </TextField>
              <Button
                variant="contained"
                fullWidth
                disabled={enrollBusy || enrollUserId === ""}
                onClick={submitEnroll}
                aria-label="Enroll user"
              >
                {enrollBusy ? "Enrolling…" : "Enroll"}
              </Button>
            </Box>
          </Drawer>

          <Drawer anchor="right" open={assignmentCourseId != null} onClose={() => setAssignmentCourseId(null)}>
            <Box sx={{ width: 500, p: 3 }} role="region" aria-label="Course assignments">
              <Typography variant="h3" sx={{ mb: 2 }}>
                Assignments — {assignmentCourse?.code ?? ""}
              </Typography>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Existing assignments
              </Typography>
              <List dense sx={{ mb: 3, border: 1, borderColor: "divider", borderRadius: 1 }}>
                {assignments.length === 0 ? (
                  <ListItem>
                    <ListItemText primary="No assignments yet" />
                  </ListItem>
                ) : (
                  assignments.map((a) => (
                    <ListItem key={a.id} alignItems="flex-start">
                      <ListItemText
                        primary={
                          <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap" }}>
                            <Typography variant="body2" fontWeight={600} component="span">
                              {a.title}
                            </Typography>
                            <Chip size="small" label={a.modality} aria-label={`Modality ${a.modality}`} />
                          </Box>
                        }
                        secondary={`Created: ${a.created_at ?? "—"}`}
                      />
                    </ListItem>
                  ))
                )}
              </List>
              {assignmentCourseId != null && (
                <AssignmentForm
                  courseId={assignmentCourseId}
                  onSuccess={async () => {
                    await loadAssignments(assignmentCourseId);
                    await refresh();
                    setSnackbar({ open: true, message: "Assignment created", severity: "success" });
                  }}
                />
              )}
            </Box>
          </Drawer>
        </Box>
      )}

      {tab === 2 && (
        <Box id="admin-panel-audit" role="tabpanel" aria-labelledby="admin-tab-audit">
          <DataGrid
            rows={audit}
            columns={auditColumns}
            autoHeight
            pageSizeOptions={[25]}
            initialState={{ pagination: { paginationModel: { pageSize: 25 } } }}
            sx={{ border: 1, borderColor: "divider", borderRadius: 2 }}
            aria-label="Audit log"
          />
        </Box>
      )}
    </Box>
  );
}
