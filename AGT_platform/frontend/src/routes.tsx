// src/routes.tsx
import React from "react";
import {
  createBrowserRouter,
  createRoutesFromElements,
  Route,
  Navigate,
} from "react-router-dom";
import { jwtDecode } from "jwt-decode";
import Login from "./pages/Login";
import StudentDashboard from "./pages/StudentDashboard";
import TeacherDashboard from "./pages/TeacherDashboard";
import AdminDashboard from "./pages/AdminDashboard";
import { getToken } from "./auth";

interface JwtPayload {
  role: string;
  [key: string]: any;
}

function PrivateWrapper({ children }: { children: JSX.Element }) {
  const token = getToken();
  if (!token) {
    return <Navigate to="/login" replace />;
  }
  return children;
}

function RoleBasedDashboard() {
  const token = getToken();
  if (!token) {
    return <Navigate to="/login" replace />;
  }
  const { role } = jwtDecode<JwtPayload>(token) || {};
  switch (role) {
    case "admin":
      return <AdminDashboard />;
    case "teacher":
      return <TeacherDashboard />;
    default:
      return <StudentDashboard />;
  }
}

// Use createRoutesFromElements to define routes declaratively.
const router = createBrowserRouter(
  createRoutesFromElements(
    <>
      {/* Login route is public */}
      <Route path="/login" element={<Login />} />
      {/* All other routes go through PrivateWrapper and render the correct dashboard */}
      <Route
        path="/*"
        element={
          <PrivateWrapper>
            <RoleBasedDashboard />
          </PrivateWrapper>
        }
      />
    </>
  )
);

export default router;
export { router };
