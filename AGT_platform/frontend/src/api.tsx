export type Assignment = {
  id: string;
  filename: string;
  status: "uploaded" | "grading" | "graded" | "failed";
  suggested_grade?: number;
  feedback?: string;
  created_at?: string;
};

const API_BASE = ""; // use Vite proxy: /api -> backend

export async function listAssignments(): Promise<Assignment[]> {
  const res = await fetch(`${API_BASE}/api/assignments`, { credentials: "include" });
  if (!res.ok) throw new Error(`listAssignments failed: ${res.status}`);
  return res.json();
}

export async function uploadAssignment(file: File): Promise<{ id: string }> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_BASE}/api/assignments`, {
    method: "POST",
    body: form,
    credentials: "include",
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`uploadAssignment failed: ${res.status} ${text}`);
  }
  return res.json();
}

export async function startGrading(id: string): Promise<{ ok: true }> {
  const res = await fetch(`${API_BASE}/api/assignments/${id}/grade`, {
    method: "POST",
    credentials: "include",
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`startGrading failed: ${res.status} ${text}`);
  }
  return res.json();
}

export async function getAssignment(id: string): Promise<Assignment> {
  const res = await fetch(`${API_BASE}/api/assignments/${id}`, { credentials: "include" });
  if (!res.ok) throw new Error(`getAssignment failed: ${res.status}`);
  return res.json();
}


