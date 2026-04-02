import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { DataGrid, type GridColDef } from "@mui/x-data-grid";
import { Box, Button, Typography } from "@mui/material";
import AssignmentTurnedInOutlined from "@mui/icons-material/AssignmentTurnedInOutlined";
import { api } from "../api";
import StatusChip from "./StatusChip";

export type TeacherSubmissionRow = {
  id: number;
  student_id?: number;
  assignment_id?: number;
  course?: string;
  student?: string;
  assignment?: string;
  submitted?: string;
  status: string;
  final_score?: number | null;
};

function NoRows() {
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        py: 6,
        color: "text.secondary",
      }}
    >
      <AssignmentTurnedInOutlined sx={{ fontSize: 56, mb: 1, opacity: 0.45 }} aria-hidden />
      <Typography variant="subtitle1" fontWeight={600}>
        No submissions yet
      </Typography>
      <Typography variant="body2" textAlign="center" maxWidth={360}>
        Students will appear here once they submit their first assignment.
      </Typography>
    </Box>
  );
}

export default function TeacherSubmissionsGrid() {
  const navigate = useNavigate();
  const [rows, setRows] = useState<TeacherSubmissionRow[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.get("/api/teacher/submissions?limit=200");
      const list = Array.isArray(data) ? data : (data as { data?: unknown }).data;
      if (Array.isArray(list)) {
        setRows(
          list.map((r: Record<string, unknown>, i: number) => ({
            id: Number(r.id ?? i),
            student_id: r.student_id != null ? Number(r.student_id) : undefined,
            assignment_id: r.assignment_id != null ? Number(r.assignment_id) : undefined,
            course: r.course != null ? String(r.course) : "—",
            student: r.student != null ? String(r.student) : String(r.student_id ?? "—"),
            assignment: r.assignment != null ? String(r.assignment) : String(r.assignment_id ?? "—"),
            submitted: r.submitted != null ? String(r.submitted) : (r.created_at != null ? String(r.created_at) : "—"),
            status: String(r.status ?? "queued"),
            final_score: r.final_score != null ? Number(r.final_score) : null,
          })),
        );
      } else {
        setRows([]);
      }
    } catch {
      setRows([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const columns: GridColDef<TeacherSubmissionRow>[] = [
    { field: "id", headerName: "#", width: 70, type: "number" },
    { field: "student", headerName: "Student", flex: 1, minWidth: 120 },
    { field: "assignment", headerName: "Assignment", flex: 1, minWidth: 140 },
    { field: "course", headerName: "Course", flex: 0.8, minWidth: 100 },
    { field: "submitted", headerName: "Submitted", flex: 1, minWidth: 160 },
    {
      field: "status",
      headerName: "Status",
      width: 140,
      renderCell: (params) => <StatusChip status={params.value as string} />,
    },
    {
      field: "final_score",
      headerName: "Score",
      width: 90,
      type: "number",
      valueFormatter: (v) => (v != null ? String(v) : "—"),
    },
    {
      field: "actions",
      headerName: "Action",
      width: 100,
      sortable: false,
      renderCell: (params) => (
        <Button
          size="small"
          variant="outlined"
          onClick={() => navigate(`/submissions/${params.id}`)}
          aria-label={`Review submission ${params.id}`}
        >
          Review
        </Button>
      ),
    },
  ];

  return (
    <Box sx={{ width: "100%", minHeight: 420 }}>
      <Box sx={{ display: "flex", justifyContent: "flex-end", mb: 1 }}>
        <Button variant="outlined" size="small" onClick={load} disabled={loading} aria-label="Refresh submissions list">
          Refresh
        </Button>
      </Box>
      <DataGrid
        rows={rows}
        columns={columns}
        loading={loading}
        pageSizeOptions={[25]}
        initialState={{ pagination: { paginationModel: { pageSize: 25 } } }}
        disableRowSelectionOnClick
        slots={{
          noRowsOverlay: NoRows,
        }}
        sx={{
          border: 1,
          borderColor: "divider",
          borderRadius: 2,
          "& .MuiDataGrid-columnHeaders": { bgcolor: "#F1F5F9", fontWeight: 700 },
        }}
        aria-label="Recent submissions"
      />
    </Box>
  );
}
