import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import reportWebVitals from "./reportWebVitals";

// 1. Keep Bootstrap CSS import
import 'bootstrap/dist/css/bootstrap.min.css';

// 2. Removed: react-router-dom imports, redux imports, store logic

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    {/* 3. Removed: Provider (Redux) and RouterProvider (Routing) */}
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
