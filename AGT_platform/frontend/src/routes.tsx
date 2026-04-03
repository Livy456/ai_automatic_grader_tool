import { useEffect, useState } from "react";
import {
  createBrowserRouter,
  createRoutesFromElements,
  Navigate,
  Outlet,
  Route,
  useLocation,
} from "react-router-dom";
import { Box, CircularProgress } from "@mui/material";
import { jwtDecode } from "jwt-decode";
import { refreshAccessToken } from "./api";
import { getToken } from "./auth";
import Shell from "./components/Shell";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import SubmitAssignment from "./pages/SubmitAssignment";
import SubmissionReview from "./pages/SubmissionReview";
import AssignmentsBrowse from "./pages/AssignmentsBrowse";
import AssignmentDetail from "./pages/AssignmentDetail";
import MyGrades from "./pages/MyGrades";
import SubmissionsList from "./pages/SubmissionsList";
import StandaloneAutograder from "./pages/StandaloneAutograder";
import StandaloneResult from "./pages/StandaloneResult";

interface JwtPayload {
  role: string;
}

function PrivateLayout() {
  const location = useLocation();
  const publicAutograder = location.pathname.startsWith("/autograder");
  const [gate, setGate] = useState<"loading" | "in" | "out">("loading");

  useEffect(() => {
    if (publicAutograder) {
      setGate("in");
      return;
    }
    let cancelled = false;
    (async () => {
      if (getToken()) {
        if (!cancelled) setGate("in");
        return;
      }
      const ok = await refreshAccessToken();
      if (!cancelled) setGate(ok ? "in" : "out");
    })();
    return () => {
      cancelled = true;
    };
  }, [publicAutograder]);

  if (publicAutograder) {
    return <Outlet />;
  }

  if (gate === "loading") {
    return (
      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          minHeight: "50vh",
        }}
      >
        <CircularProgress aria-label="Loading session" />
      </Box>
    );
  }
  if (gate === "out") {
    return <Navigate to="/login" replace />;
  }
  return <Outlet />;
}

function TeacherRedirect() {
  const token = getToken();
  if (!token) return <Navigate to="/login" replace />;
  let role = "student";
  try {
    role = jwtDecode<JwtPayload>(token).role;
  } catch {
    /* ignore */
  }
  if (role !== "teacher" && role !== "admin") {
    return <Navigate to="/" replace />;
  }
  return <Navigate to="/submissions" replace />;
}

const router = createBrowserRouter(
  createRoutesFromElements(
    <>
      <Route path="/login" element={<Login />} />
      <Route element={<PrivateLayout />}>
        <Route element={<Shell />}>
          <Route index element={<Dashboard />} />
          <Route path="admin" element={<Dashboard />} />
          <Route path="teacher" element={<TeacherRedirect />} />
          <Route path="grades" element={<MyGrades />} />
          <Route path="assignments" element={<AssignmentsBrowse />} />
          <Route path="assignments/:id" element={<AssignmentDetail />} />
          <Route path="assignments/:id/submit" element={<SubmitAssignment />} />
          <Route path="submissions" element={<SubmissionsList />} />
          <Route path="submissions/:id" element={<SubmissionReview />} />
          <Route path="autograder" element={<StandaloneAutograder />} />
          <Route path="autograder/:id" element={<StandaloneResult />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Route>
    </>
  ),
);

export default router;
export { router };
