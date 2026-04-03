// src/auth.ts — access JWT lives in memory only (not localStorage) to reduce XSS persistence.
let memoryAccessToken: string | null = null;

export function getToken(): string | null {
  return memoryAccessToken;
}

export function setToken(t: string) {
  memoryAccessToken = t;
}

export function clearToken() {
  memoryAccessToken = null;
}
