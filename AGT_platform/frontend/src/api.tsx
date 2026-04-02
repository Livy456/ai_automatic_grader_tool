/// <reference types="vite/client" />
import { getToken } from "./auth";

function withAuthHeaders(extra?: HeadersInit): Headers {
  const h = new Headers(extra ?? {});
  const token = getToken();
  if (token) h.set("Authorization", `Bearer ${token}`);
  return h;
}

export type Assignment = {
  id: string;
  filename: string;
  status: string;
  suggested_grade?: number | null;
  feedback?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

/**
 * Browser → S3 (presigned PUT). Do not send auth headers; signature embeds access.
 */
export async function putToPresignedUrl(
  url: string,
  body: Blob,
  contentType: string
): Promise<void> {
  const res = await fetch(url, {
    method: "PUT",
    body,
    headers: { "Content-Type": contentType },
  });
  if (!res.ok) {
    const t = await res.text().catch(() => "");
    throw new Error(`S3 PUT failed: ${res.status} ${t}`);
  }
}

export type DirectUploadStartResponse = {
  submission_id: number;
  status: string;
  uploads: Array<{
    artifact_id: number;
    s3_key: string;
    upload_url: string;
    content_type: string;
  }>;
};

/**
 * Presigned flow: start → PUT each file to S3 → finalize. Keeps large files off the API EC2.
 */
export async function submitAssignmentDirect(
  assignmentId: number,
  files: File[],
  onProgress?: (fileIndex: number, fraction: number) => void
): Promise<{ submission_id: number; status: string }> {
  const start = (await api.post("/api/submissions/direct-upload/start", {
    assignment_id: assignmentId,
    files: files.map((f) => ({
      filename: f.name,
      content_type: f.type || "application/octet-stream",
    })),
  })) as DirectUploadStartResponse;

  for (let i = 0; i < start.uploads.length; i++) {
    const u = start.uploads[i];
    const file = files[i];
    if (!file) continue;
    await putToPresignedUrl(u.upload_url, file, u.content_type);
    onProgress?.(i, 1);
  }

  const done = (await api.post(`/api/submissions/${start.submission_id}/finalize`, {})) as {
    submission_id: number;
    status: string;
  };
  return done;
}

export const api = {
  async get(path: string) {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "GET",
      headers: withAuthHeaders(),
      credentials: "include",
    });
    const text = await res.text().catch(() => "");
    if (!res.ok) throw new Error(`GET ${path} failed: ${res.status} ${text}`);
    return text ? JSON.parse(text) : null;
  },

  async post(path: string, body?: BodyInit | object) {
    const headers = withAuthHeaders();
    let payload: BodyInit | undefined;
    if (body !== undefined && body !== null && typeof body === "object" && !(body instanceof FormData)) {
      payload = JSON.stringify(body);
      headers.set("Content-Type", "application/json");
    } else {
      payload = body as BodyInit;
      if (typeof body === "string") headers.set("Content-Type", "application/json");
      if (body instanceof FormData) headers.delete("Content-Type");
    }

    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      body: payload,
      headers,
      credentials: "include",
    });

    const text = await res.text().catch(() => "");
    if (!res.ok) throw new Error(`POST ${path} failed: ${res.status} ${text}`);
    return text ? JSON.parse(text) : null;
  },

  async patch(path: string, body: object) {
    const headers = withAuthHeaders();
    headers.set("Content-Type", "application/json");
    const res = await fetch(`${API_BASE}${path}`, {
      method: "PATCH",
      body: JSON.stringify(body),
      headers,
      credentials: "include",
    });
    const text = await res.text().catch(() => "");
    if (!res.ok) throw new Error(`PATCH ${path} failed: ${res.status} ${text}`);
    return text ? JSON.parse(text) : null;
  },

  async delete(path: string) {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "DELETE",
      headers: withAuthHeaders(),
      credentials: "include",
    });
    const text = await res.text().catch(() => "");
    if (!res.ok) throw new Error(`DELETE ${path} failed: ${res.status} ${text}`);
    return text ? JSON.parse(text) : null;
  },
};

export async function listAssignments(): Promise<Assignment[]> {
  return api.get("/api/assignments");
}

export async function uploadAssignment(file: File): Promise<{ id: string }> {
  const fd = new FormData();
  fd.append("file", file);
  return api.post("/api/assignments", fd);
}

export async function getAssignment(id: string): Promise<Assignment> {
  return api.get(`/api/assignments/${id}`);
}

export async function startGrading(id: string): Promise<{ ok: boolean }> {
  return api.post(`/api/assignments/${id}/grade`);
}

// ── Course / Enrollment / Assignment types ────────────────────────────────

export type CourseListItem = {
  id: number;
  code: string;
  title: string;
  enrollment_role: "student" | "teacher" | null;
};

export type CourseDetail = {
  id: number;
  code: string;
  title: string;
  enrollments: Array<{
    enrollment_id?: number;
    user_id: number;
    email: string;
    name: string;
    role: "student" | "teacher";
  }>;
};

export type RubricCriterion = {
  criterion: string;
  max_score: number;
};

export type CourseAssignment = {
  id: number;
  course_id: number;
  title: string;
  description: string;
  modality: string;
  rubric: RubricCriterion[];
  due_date: string | null;
  created_at: string | null;
};

export type CreateAssignmentPayload = {
  title: string;
  description?: string;
  modality: string;
  rubric?: RubricCriterion[];
  due_date?: string | null;
};

export type CreateEnrollmentPayload = {
  course_id: number;
  user_id: number;
  role: "student" | "teacher";
};

// ── Course helpers ────────────────────────────────────────────────────────

export function listCourses(): Promise<CourseListItem[]> {
  return api.get("/api/courses");
}

export function getCourse(courseId: number): Promise<CourseDetail> {
  return api.get(`/api/courses/${courseId}`);
}

export function listCourseAssignments(courseId: number): Promise<CourseAssignment[]> {
  return api.get(`/api/courses/${courseId}/assignments`);
}

export function createCourseAssignment(
  courseId: number,
  payload: CreateAssignmentPayload
): Promise<{ id: number; title: string; course_id: number }> {
  return api.post(`/api/courses/${courseId}/assignments`, payload);
}

export function updateCourseAssignment(
  courseId: number,
  assignmentId: number,
  payload: Partial<CreateAssignmentPayload>
): Promise<{ id: number; title: string }> {
  return api.patch(`/api/courses/${courseId}/assignments/${assignmentId}`, payload);
}

// ── Admin enrollment helpers ──────────────────────────────────────────────

export function adminListCourseEnrollments(
  courseId: number
): Promise<CourseDetail["enrollments"]> {
  return api.get(`/api/admin/courses/${courseId}/enrollments`);
}

export function adminEnrollUser(payload: CreateEnrollmentPayload): Promise<{ id: number }> {
  return api.post("/api/admin/enrollments", payload);
}

export function adminRemoveEnrollment(enrollmentId: number): Promise<{ ok: boolean }> {
  return api.delete(`/api/admin/enrollments/${enrollmentId}`);
}
