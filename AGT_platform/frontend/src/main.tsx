// src/main.tsx
import React from "react";
import ReactDOM from "react-dom/client";
import { RouterProvider } from "react-router-dom";
import router from "./routes"; // default import of the Router instance

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    {/* Pass the router object to RouterProvider */}
    <RouterProvider router={router} />
  </React.StrictMode>
);
