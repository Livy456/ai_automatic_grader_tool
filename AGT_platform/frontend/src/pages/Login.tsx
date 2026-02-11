/// <reference types="vite/client" />
import { useEffect, useMemo, useState } from "react";
import { setToken } from "../auth";
import { Container, TextField, Typography, Box, Alert, Divider, Button } from "@mui/material";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:5000";

type DiscoverResponse =
  | { supported: true; domain: string }
  | { supported: false; domain: string; message?: string }
  | { error: string };

export default function Login() {
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  // If backend returned token in fragment
  useEffect(() => {
    const hash = new URLSearchParams(window.location.hash.replace("#", ""));
    const t = hash.get("token");
    if (t) {
      setToken(t);
      window.location.href = "/"; // go to router landing
    }
    
    // Check for error in query params
    const params = new URLSearchParams(window.location.search);
    const error = params.get("error");
    if (error) {
      setErr(error);
    }
  }, []);

  const canSubmit = useMemo(() => email.includes("@") && email.length > 5, [email]);

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
        // Start OIDC flow
        window.location.href = `${API_BASE}/api/auth/login?email=${encodeURIComponent(email)}`;
        return;
      }

      if ("supported" in data && !data.supported) {
        setInfo(data.message ?? "Your school is not configured yet.");
        return;
      }

      throw new Error("Unexpected response from server.");
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  }

  function handleMicrosoftLogin() {
    window.location.href = `${API_BASE}/api/auth/login/microsoft`;
  }

  function handleGoogleLogin() {
    window.location.href = `${API_BASE}/api/auth/login/google`;
  }

  return (
    <Container maxWidth="sm" sx={{ mt: 10 }}>
      <Typography variant="h4" gutterBottom>
        AI Grader
      </Typography>

      <Typography sx={{ mb: 3, color: "text.secondary" }}>
        Sign in with your college account. Use your institution's SSO or sign in with Microsoft or Google using your college email.
      </Typography>

      <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {/* OAuth Provider Buttons */}
        <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <Button
            variant="outlined"
            fullWidth
            onClick={handleMicrosoftLogin}
            sx={{
              textTransform: "none",
              py: 1.5,
              borderColor: "#0078d4",
              color: "#0078d4",
              fontWeight: 500,
              "&:hover": {
                borderColor: "#005a9e",
                backgroundColor: "rgba(0, 120, 212, 0.04)",
              },
            }}
          >
            <Box
              component="svg"
              sx={{
                width: 20,
                height: 20,
                mr: 1.5,
              }}
              viewBox="0 0 23 23"
              fill="none"
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
            sx={{
              textTransform: "none",
              py: 1.5,
              borderColor: "#4285f4",
              color: "#4285f4",
              fontWeight: 500,
              "&:hover": {
                borderColor: "#3367d6",
                backgroundColor: "rgba(66, 133, 244, 0.04)",
              },
            }}
          >
            <Box
              component="svg"
              sx={{
                width: 20,
                height: 20,
                mr: 1.5,
              }}
              viewBox="0 0 24 24"
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

        <Divider sx={{ my: 2 }}>
          <Typography variant="body2" color="text.secondary">
            OR
          </Typography>
        </Divider>

        {/* Institution SSO */}
        <TextField
          label="College email"
          placeholder="name@university.edu"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          autoComplete="email"
          fullWidth
        />

        <Button
          variant="contained"
          disabled={!canSubmit || busy}
          onClick={handleContinue}
        >
          {busy ? "Checking..." : "Continue with Institution SSO"}
        </Button>

        {err && <Alert severity="error">{err}</Alert>}
        {info && <Alert severity="info">{info}</Alert>}

        <Typography variant="body2" sx={{ mt: 2, color: "text.secondary", textAlign: "center" }}>
          Please use your college or university email address to sign in.
        </Typography>
      </Box>
    </Container>
  );
}
