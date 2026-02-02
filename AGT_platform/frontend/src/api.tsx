/// <reference types="vite/client" />
// src/api.tsx (or src/api.ts)
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
 * api: simple wrapper used by dashboards.
 * Usage: api.get("/api/assignments"), api.post("/api/assignments", formData)
 */
export const api = {
  async get(path: string) {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "GET",
      headers: withAuthHeaders(),
      // credentials optional; keep if you want cookies too
      credentials: "include",
    });
    const text = await res.text().catch(() => "");
    if (!res.ok) throw new Error(`GET ${path} failed: ${res.status} ${text}`);
    return text ? JSON.parse(text) : null;
  },

  async post(path: string, body?: BodyInit) {
    const headers = withAuthHeaders();

    // If it's JSON string, declare content-type
    if (typeof body === "string") headers.set("Content-Type", "application/json");
    // If it's FormData, DO NOT set Content-Type (browser sets boundary)
    if (body instanceof FormData) headers.delete("Content-Type");

    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      body,
      headers,
      credentials: "include",
    });

    const text = await res.text().catch(() => "");
    if (!res.ok) throw new Error(`POST ${path} failed: ${res.status} ${text}`);
    return text ? JSON.parse(text) : null;
  },
};

// Convenience functions (optional, nice for pages)
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