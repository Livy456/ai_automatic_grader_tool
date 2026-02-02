import { useEffect, useState } from "react";
import { api } from "../api";
import { Container, Typography, Box, Button, TextField, MenuItem } from "@mui/material";
import DataTable from "../components/DataTable";

export default function AdminDashboard() {
  const [users, setUsers] = useState<any[]>([]);
  const [audit, setAudit] = useState<any[]>([]);
  const [courses, setCourses] = useState<any[]>([]);
  const [newCourseCode, setNewCourseCode] = useState("");
  const [newCourseTitle, setNewCourseTitle] = useState("");

  const refresh = async () => {
    const [u, a, c] = await Promise.all([
      api.get("/admin/users"),
      api.get("/admin/audit"),
      api.get("/admin/courses")
    ]);
    setUsers(u.data);
    setAudit(a.data);
    setCourses(c.data);
  };

  useEffect(() => { refresh(); }, []);

  const setRole = async (userId: number, role: string) => {
    await api.post(`/admin/users/${userId}/role`, { role } as any);
    await refresh();
  };

  const createCourse = async () => {
    await api.post("/admin/courses", { code: newCourseCode, title: newCourseTitle } as any);
    setNewCourseCode(""); setNewCourseTitle("");
    await refresh();
  };

  return (
    <Container>
      <Typography variant="h5" gutterBottom>Admin Dashboard</Typography>

      <Typography variant="h6" sx={{ mt: 3 }}>Courses</Typography>
      <DataTable
        columns={[
          { key: "id", label: "ID" },
          { key: "code", label: "Code" },
          { key: "title", label: "Title" }
        ]}
        rows={courses}
      />
      <Box sx={{ display: "flex", gap: 2, mt: 2 }}>
        <TextField label="Course Code" value={newCourseCode} onChange={(e) => setNewCourseCode(e.target.value)} />
        <TextField label="Course Title" value={newCourseTitle} onChange={(e) => setNewCourseTitle(e.target.value)} fullWidth />
        <Button variant="contained" onClick={createCourse}>Create</Button>
      </Box>

      <Typography variant="h6" sx={{ mt: 4 }}>Users</Typography>
      <DataTable
        columns={[
          { key: "id", label: "ID" },
          { key: "email", label: "Email" },
          { key: "role", label: "Role" }
        ]}
        rows={users}
        onRowClick={(u) => {
          const role = prompt("Set role to student|teacher|admin", u.role);
          if (role) setRole(u.id, role);
        }}
      />
      <Typography sx={{ mt: 1, fontSize: 13, color: "#666" }}>
        Tip: click a user row to change role.
      </Typography>

      <Typography variant="h6" sx={{ mt: 4 }}>Audit Log</Typography>
      <DataTable
        columns={[
          { key: "time", label: "Time" },
          { key: "actor_user_id", label: "Actor" },
          { key: "action", label: "Action" },
          { key: "target_type", label: "Target" },
          { key: "target_id", label: "Target ID" }
        ]}
        rows={audit}
      />

      <Box sx={{ mt: 2 }}>
        <Button variant="outlined" onClick={refresh}>Refresh</Button>
      </Box>
    </Container>
  );
}
