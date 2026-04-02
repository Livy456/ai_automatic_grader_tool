import { createTheme } from "@mui/material/styles";

const primary = "#1A1F5E";
const secondary = "#4F46E5";
const accent = "#06B6D4";
const success = "#10B981";
const warning = "#F59E0B";
const error = "#EF4444";
const bg = "#F8FAFC";
const surface = "#FFFFFF";
const border = "#E2E8F0";
const textPrimary = "#0F172A";
const textSecondary = "#64748B";

export const appTheme = createTheme({
  palette: {
    mode: "light",
    primary: { main: primary, contrastText: "#FFFFFF" },
    secondary: { main: secondary, contrastText: "#FFFFFF" },
    success: { main: success },
    warning: { main: warning },
    error: { main: error },
    background: { default: bg, paper: surface },
    text: { primary: textPrimary, secondary: textSecondary },
    divider: border,
  },
  typography: {
    fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
    h1: { fontSize: "2rem", fontWeight: 700, lineHeight: 1.2 },
    h2: { fontSize: "1.5rem", fontWeight: 600, lineHeight: 1.3 },
    h3: { fontSize: "1.25rem", fontWeight: 600, lineHeight: 1.35 },
    body1: { fontSize: "0.875rem", fontWeight: 400 },
    body2: { fontSize: "0.875rem", fontWeight: 400 },
    caption: { fontSize: "0.75rem", fontWeight: 400 },
    button: { textTransform: "none", fontWeight: 600 },
  },
  shape: { borderRadius: 12 },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: { backgroundColor: bg },
        "*, *::before, *::after": {
          outlineColor: `${secondary}80`,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: "none",
          border: `1px solid ${border}`,
          borderRadius: 12,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: { borderRadius: 8, textTransform: "none", fontWeight: 600 },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: { borderRadius: 6 },
      },
    },
    MuiTextField: {
      defaultProps: { variant: "outlined" },
      styleOverrides: {
        root: {
          "& .MuiOutlinedInput-root": { borderRadius: 8 },
        },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: { borderRadius: 8 },
      },
    },
    MuiTableHead: {
      styleOverrides: {
        root: {
          backgroundColor: "#F1F5F9",
          "& .MuiTableCell-head": { fontWeight: 700 },
        },
      },
    },
  },
});

export const brandAccent = accent;
