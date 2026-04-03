/// <reference types="vite/client" />
import { useEffect, useMemo, useState } from "react";
import { refreshAccessToken } from "../api";
import { setToken } from "../auth";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Collapse,
  Divider,
  TextField,
  Typography,
} from "@mui/material";

function trimApiBase(v: unknown): string {
  if (v == null) return "";
  const s = String(v).trim();
  return s.replace(/\/+$/, "");
}

const API_BASE =
  trimApiBase(import.meta.env.VITE_API_BASE) || "http://localhost:5000";

const OAUTH_API_ORIGIN: string = (() => {
  const explicit = trimApiBase(import.meta.env.VITE_OAUTH_API_ORIGIN);
  if (explicit) return explicit;
  if (/:(5173|5174)(\/|$)/.test(API_BASE)) {
    return "http://localhost:5000";
  }
  return API_BASE;
})();

type DiscoverResponse =
  | { supported: true; domain: string }
  | { supported: false; domain: string; message?: string }
  | { error: string };

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [pwBusy, setPwBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [pwOpen, setPwOpen] = useState(false);

  useEffect(() => {
    const hash = new URLSearchParams(window.location.hash.replace("#", ""));
    const t = hash.get("token");
    if (t) {
      setToken(t);
      window.location.replace("/");
      return;
    }

    const params = new URLSearchParams(window.location.search);
    const error = params.get("error");
    if (error) {
      try {
        setErr(decodeURIComponent(error));
      } catch {
        setErr(error);
      }
    }

    const reason = params.get("reason");
    if (reason === "session_expired") {
      setErr("Your session has expired. Please sign in again to continue.");
    }

    if (!error) {
      void refreshAccessToken().then((ok) => {
        if (ok) window.location.replace("/");
      });
    }
  }, []);

  const canSubmit = useMemo(() => email.includes("@") && email.length > 5, [email]);
  const canPasswordLogin = useMemo(
    () => email.includes("@") && password.length >= 1,
    [email, password],
  );

  async function handlePasswordLogin() {
    setErr(null);
    setInfo(null);
    setPwBusy(true);
    try {
      const res = await fetch(`${API_BASE}/api/auth/login/password`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ email: email.trim().toLowerCase(), password }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error((data as { error?: string }).error || `login failed (${res.status})`);
      }
      const token = (data as { access_token?: string }).access_token;
      if (!token) throw new Error("no access_token in response");
      setToken(token);
      window.location.href = "/";
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setPwBusy(false);
    }
  }

  async function handleContinue() {
    setErr(null);
    setInfo(null);
    setBusy(true);
    try {
      const res = await fetch(`${API_BASE}/api/auth/discover`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ email }),
        credentials: "include",
      });

      const data = (await res.json()) as DiscoverResponse;

      if (!res.ok) {
        throw new Error("error" in data ? data.error : `discover failed (${res.status})`);
      }

      if ("supported" in data && data.supported) {
        window.location.href = `${OAUTH_API_ORIGIN}/api/auth/login?email=${encodeURIComponent(email)}`;
        return;
      }

      if ("supported" in data && !data.supported) {
        setInfo(data.message ?? "Your school is not configured yet.");
        return;
      }

      throw new Error("Unexpected response from server.");
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  function handleMicrosoftLogin() {
    window.location.href = `${OAUTH_API_ORIGIN}/api/auth/login/microsoft`;
  }

  function handleGoogleLogin() {
    window.location.href = `${OAUTH_API_ORIGIN}/api/auth/login/google`;
  }

  return (
    <Box
      sx={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        px: 2,
        py: 4,
        bgcolor: "background.default",
      }}
    >
      <Box sx={{ width: "100%", maxWidth: 440 }}>
        <Box sx={{ textAlign: "center", mb: 3 }}>
          <Typography variant="h1" sx={{ color: "primary.main", fontWeight: 700, mb: 1 }}>
            AI Grader
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Intelligent grading for modern education
          </Typography>
        </Box>

        {err && (
          <Alert severity="error" sx={{ mb: 2 }} role="alert">
            {err}
          </Alert>
        )}
        {info && (
          <Alert severity="info" sx={{ mb: 2 }}>
            {info}
          </Alert>
        )}

        <Card elevation={0}>
          <CardContent sx={{ p: 3 }}>
            <Typography variant="h3" sx={{ mb: 2 }}>
              Sign in to continue
            </Typography>

            <TextField
              label="Email address"
              placeholder="name@university.edu"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
              fullWidth
              sx={{ mb: 2 }}
              aria-label="Email address"
            />

            <Button
              variant="contained"
              color="secondary"
              fullWidth
              size="large"
              disabled={!canSubmit || busy}
              onClick={handleContinue}
              aria-label="Continue with institution SSO"
            >
              {busy ? <CircularProgress size={20} color="inherit" aria-label="Loading" /> : "Continue →"}
            </Button>

            <Divider sx={{ my: 3 }}>
              <Typography variant="caption" color="text.secondary">
                or
              </Typography>
            </Divider>

            <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
              <Button
                variant="outlined"
                fullWidth
                onClick={handleMicrosoftLogin}
                aria-label="Sign in with Microsoft"
                sx={{ py: 1.25, borderColor: "#0078d4", color: "#0078d4" }}
              >
                <Box
                  component="svg"
                  sx={{ width: 20, height: 20, mr: 1.5 }}
                  viewBox="0 0 23 23"
                  fill="none"
                  aria-hidden
                >
                  <path
                    d="M11.4 11.4H0V0h11.4v11.4zM23 11.4H11.6V0H23v11.4zM11.4 23H0V11.6h11.4V23zM23 23H11.6V11.6H23V23z"
                    fill="#0078d4"
                  />
                </Box>
                Sign in with Microsoft
              </Button>

              <Button
                variant="outlined"
                fullWidth
                onClick={handleGoogleLogin}
                aria-label="Sign in with Google"
                sx={{ py: 1.25, borderColor: "#4285f4", color: "#4285f4" }}
              >
                <Box
                  component="svg"
                  sx={{ width: 20, height: 20, mr: 1.5 }}
                  viewBox="0 0 24 24"
                  aria-hidden
                >
                  <path
                    fill="#4285F4"
                    d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                  />
                  <path
                    fill="#34A853"
                    d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                  />
                  <path
                    fill="#FBBC05"
                    d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
                  />
                  <path
                    fill="#EA4335"
                    d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
                  />
                </Box>
                Sign in with Google
              </Button>
            </Box>

            <Divider sx={{ my: 3 }}>
              <Typography variant="caption" color="text.secondary" align="center" display="block">
                Password login (for admins)
              </Typography>
            </Divider>

            <Button
              size="small"
              onClick={() => setPwOpen((o) => !o)}
              aria-expanded={pwOpen}
              aria-controls="login-password-section"
              sx={{ mb: 1 }}
            >
              {pwOpen ? "Hide" : "Show"} password login
            </Button>

            <Collapse in={pwOpen}>
              <Box id="login-password-section" sx={{ pt: 1 }}>
                <TextField
                  label="Password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete="current-password"
                  fullWidth
                  sx={{ mb: 2 }}
                  aria-label="Password"
                />
                <Button
                  variant="contained"
                  color="primary"
                  disabled={!canPasswordLogin || pwBusy}
                  onClick={handlePasswordLogin}
                  fullWidth
                  aria-label="Sign in with password"
                >
                  {pwBusy ? <CircularProgress size={20} color="inherit" aria-label="Signing in" /> : "Sign in"}
                </Button>
              </Box>
            </Collapse>
          </CardContent>
        </Card>

        <Typography variant="body2" color="text.secondary" sx={{ mt: 3, textAlign: "center" }}>
          College or university email required.
        </Typography>
      </Box>
    </Box>
  );
}
