import { createBrowserRouter } from "react-router-dom";
import App from "./App";
import Login from "./pages/Login";
import StudentDashboard from "./pages/StudentDashboard";
import TeacherDashboard from "./pages/TeacherDashboard";
import AdminDashboard from "./pages/AdminDashboard";
import AssignmentDetail from "./pages/AssignmentDetail";
import SubmissionUpload from "./pages/SubmissionUpload";
import SubmissionReview from "./pages/SubmissionReview";
import AssignmentsPage from "./pages/AssignmentsPage";

export const router = createBrowserRouter([
  { path: "/login", element: <Login /> },

  {
    path: "/",
    element: <App />,
    children: [
      
      { path: "/", element: <StudentDashboard /> },
      { path: "/assignment", element: <AssignmentsPage /> },
      { path: "/teacher", element: <TeacherDashboard /> },
      { path: "/admin", element: <AdminDashboard /> },

      { path: "/assignments/:id", element: <AssignmentDetail /> },
      { path: "/assignments/:id/submit", element: <SubmissionUpload /> },
      { path: "/submissions/:id", element: <SubmissionReview /> },
    ]
  }
]);
