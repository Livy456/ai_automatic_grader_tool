import { useMemo, useState, type ReactNode } from "react";
import {
  Outlet,
  useLocation,
  useNavigate,
  NavLink,
} from "react-router-dom";
import { jwtDecode } from "jwt-decode";
import {
  AppBar,
  Avatar,
  Box,
  Chip,
  Divider,
  Drawer,
  IconButton,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Tooltip,
  Typography,
  useMediaQuery,
  useTheme,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import LogoutOutlined from "@mui/icons-material/LogoutOutlined";
import DashboardOutlined from "@mui/icons-material/DashboardOutlined";
import GradeOutlined from "@mui/icons-material/GradeOutlined";
import UploadFileOutlined from "@mui/icons-material/UploadFileOutlined";
import AssignmentTurnedInOutlined from "@mui/icons-material/AssignmentTurnedInOutlined";
import AssignmentOutlined from "@mui/icons-material/AssignmentOutlined";
import AdminPanelSettingsOutlined from "@mui/icons-material/AdminPanelSettingsOutlined";
import { getToken, clearToken } from "../auth";

interface JwtPayload {
  id: number;
  email: string;
  role: string;
  name?: string;
}

function initialsFromPayload(p: JwtPayload): string {
  if (p.name && p.name.trim()) {
    const parts = p.name.trim().split(/\s+/).filter(Boolean);
    const a = parts[0]?.[0] ?? "";
    const b = parts[1]?.[0] ?? "";
    return (a + b).toUpperCase() || "?";
  }
  const e = p.email || "?";
  const local = e.split("@")[0] ?? e;
  if (local.length >= 2) return local.slice(0, 2).toUpperCase();
  return (e[0] ?? "?").toUpperCase();
}

function pageTitle(pathname: string): string {
  if (pathname === "/") return "Dashboard";
  if (pathname === "/login") return "Sign in";
  if (pathname === "/grades") return "My Grades";
  if (pathname === "/assignments") return "Assignments";
  if (pathname === "/submissions") return "Submissions";
  if (pathname === "/admin") return "Admin Panel";
  if (pathname === "/teacher") return "Teacher";
  const submit = pathname.match(/^\/assignments\/(\d+)\/submit$/);
  if (submit) return "Submit Assignment";
  const sub = pathname.match(/^\/submissions\/(\d+)$/);
  if (sub) return "Submission Review";
  const ad = pathname.match(/^\/assignments\/([^/]+)$/);
  if (ad) return "Assignment";
  return "AI Grader";
}

type NavItem = {
  label: string;
  to: string;
  icon: ReactNode;
  roles: Array<"student" | "teacher" | "admin"> | "all";
};

const NAV_ITEMS: NavItem[] = [
  { label: "Dashboard", to: "/", icon: <DashboardOutlined />, roles: "all" },
  { label: "My Grades", to: "/grades", icon: <GradeOutlined />, roles: ["student"] },
  {
    label: "Submit Assignment",
    to: "/assignments",
    icon: <UploadFileOutlined />,
    roles: ["student"],
  },
  {
    label: "Submissions",
    to: "/submissions",
    icon: <AssignmentTurnedInOutlined />,
    roles: ["teacher", "admin"],
  },
  {
    label: "Assignments",
    to: "/assignments",
    icon: <AssignmentOutlined />,
    roles: ["teacher", "admin"],
  },
  {
    label: "Admin Panel",
    to: "/admin",
    icon: <AdminPanelSettingsOutlined />,
    roles: ["admin"],
  },
];

function filterNav(role: string): NavItem[] {
  return NAV_ITEMS.filter((item) => {
    if (item.roles === "all") return true;
    return item.roles.includes(role as "student" | "teacher" | "admin");
  });
}

export default function Shell() {
  const theme = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const isMdUp = useMediaQuery(theme.breakpoints.up("md"));
  const isLgUp = useMediaQuery(theme.breakpoints.up("lg"));
  const [mobileOpen, setMobileOpen] = useState(false);

  const narrowNav = isMdUp && !isLgUp;
  const drawerWidth = narrowNav ? 48 : 240;

  const payload = useMemo(() => {
    const t = getToken();
    if (!t) return null;
    try {
      return jwtDecode<JwtPayload>(t);
    } catch {
      return null;
    }
  }, [location.pathname]);

  const role = payload?.role ?? "student";
  const navItems = filterNav(role);
  const title = pageTitle(location.pathname);

  const roleChipColor =
    role === "admin" ? "secondary" : role === "teacher" ? "primary" : "default";

  const drawer = (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <Toolbar
        sx={{
          minHeight: 64,
          justifyContent: narrowNav ? "center" : "flex-start",
          px: narrowNav ? 0 : 2,
        }}
        aria-label="Application sidebar header"
      >
        {!narrowNav && (
          <Typography
            variant="h6"
            component="div"
            sx={{ fontWeight: 700, color: "primary.main", fontSize: "1.1rem" }}
          >
            AI Grader
          </Typography>
        )}
      </Toolbar>
      <Divider />
      <List
        component="nav"
        sx={{ flex: 1, py: 1 }}
        aria-label="Main navigation"
      >
        {navItems.map((item) => (
          <Tooltip
            key={item.to + item.label}
            title={narrowNav ? item.label : ""}
            placement="right"
          >
            <ListItemButton
              component={NavLink}
              to={item.to}
              end={item.to === "/" ? true : undefined}
              onClick={() => setMobileOpen(false)}
              sx={{
                minHeight: 48,
                px: narrowNav ? 1.5 : 2,
                justifyContent: narrowNav ? "center" : "flex-start",
                borderLeft: "3px solid transparent",
                "&.active": {
                  borderLeftColor: "secondary.main",
                  bgcolor: "rgba(79, 70, 229, 0.08)",
                  "& .MuiListItemIcon-root": { color: "secondary.main" },
                  "& .MuiTypography-root": { fontWeight: 600 },
                },
                "& .MuiListItemIcon-root": { color: "text.secondary" },
                "&.Mui-focusVisible": { outline: "2px solid", outlineOffset: 2 },
              }}
              aria-label={item.label}
            >
              <ListItemIcon
                sx={{
                  minWidth: narrowNav ? 0 : 40,
                  justifyContent: "center",
                }}
              >
                {item.icon}
              </ListItemIcon>
              {!narrowNav && <ListItemText primary={item.label} />}
            </ListItemButton>
          </Tooltip>
        ))}
      </List>
      <Box sx={{ p: narrowNav ? 0.5 : 2, borderTop: 1, borderColor: "divider" }}>
        {!narrowNav && (
          <>
            <Typography variant="caption" color="text.secondary" display="block" noWrap title={payload?.email}>
              {payload?.email ?? ""}
            </Typography>
            <Typography variant="caption" color="text.disabled">
              v0.1
            </Typography>
          </>
        )}
        {narrowNav && (
          <Typography variant="caption" color="text.disabled" sx={{ display: "block", textAlign: "center" }}>
            v0.1
          </Typography>
        )}
      </Box>
    </Box>
  );

  function handleLogout() {
    clearToken();
    navigate("/login", { replace: true });
  }

  return (
    <Box sx={{ display: "flex", minHeight: "100vh", bgcolor: "background.default" }}>
      <AppBar
        position="fixed"
        elevation={0}
        sx={{
          zIndex: (t) => t.zIndex.drawer + 1,
          bgcolor: "background.paper",
          color: "text.primary",
          borderBottom: 1,
          borderColor: "divider",
        }}
      >
        <Toolbar sx={{ minHeight: 64 }}>
          {!isMdUp && (
            <>
              <IconButton
                color="inherit"
                edge="start"
                onClick={() => setMobileOpen(true)}
                sx={{ mr: 1 }}
                aria-label="Open navigation menu"
              >
                <MenuIcon />
              </IconButton>
              <Typography
                variant="h6"
                sx={{ fontWeight: 700, color: "primary.main", mr: 1 }}
                component="span"
              >
                AI Grader
              </Typography>
            </>
          )}
          {isMdUp && (
            <Typography
              variant="h6"
              sx={{ fontWeight: 700, color: "primary.main", mr: 2, minWidth: narrowNav ? 48 : 120 }}
            >
              {narrowNav ? "AG" : "AI Grader"}
            </Typography>
          )}
          <Typography
            component="h1"
            variant="h2"
            sx={{
              flex: 1,
              textAlign: "center",
              fontSize: { xs: "1.1rem", sm: "1.25rem" },
              fontWeight: 600,
            }}
          >
            {title}
          </Typography>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, ml: 1 }}>
            <Tooltip title={payload?.email ?? "User"}>
              <Avatar
                sx={{ width: 36, height: 36, bgcolor: "secondary.main", fontSize: "0.85rem" }}
                aria-label={`User initials ${initialsFromPayload(payload ?? { id: 0, email: "", role: "student" })}`}
              >
                {payload ? initialsFromPayload(payload) : "?"}
              </Avatar>
            </Tooltip>
            <Chip
              size="small"
              label={role}
              color={roleChipColor}
              variant={role === "student" ? "outlined" : "filled"}
              aria-label={`Role: ${role}`}
            />
            <IconButton
              color="inherit"
              onClick={handleLogout}
              aria-label="Log out"
            >
              <LogoutOutlined />
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>

      <Box component="nav" sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}>
        {!isMdUp ? (
          <Drawer
            variant="temporary"
            open={mobileOpen}
            onClose={() => setMobileOpen(false)}
            ModalProps={{ keepMounted: true }}
            sx={{
              display: { xs: "block", md: "none" },
              "& .MuiDrawer-paper": { boxSizing: "border-box", width: 240 },
            }}
          >
            {drawer}
          </Drawer>
        ) : (
          <Drawer
            variant="permanent"
            open
            sx={{
              display: { xs: "none", md: "block" },
              "& .MuiDrawer-paper": {
                boxSizing: "border-box",
                width: drawerWidth,
                borderRight: 1,
                borderColor: "divider",
              },
            }}
          >
            {drawer}
          </Drawer>
        )}
      </Box>

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          width: { md: `calc(100% - ${drawerWidth}px)` },
          minHeight: "100vh",
        }}
      >
        <Toolbar />
        <Box sx={{ p: { xs: 2, md: 3 }, maxWidth: 1400, mx: "auto" }}>
          <Outlet />
        </Box>
      </Box>
    </Box>
  );
}
