import { createRoot } from "react-dom/client";
import App from "./App.tsx";
import "./index.css";

// Register Service Worker for offline support
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").then(
      (registration) => {
        console.log("✅ Service Worker registered - App läuft jetzt offline!");
        console.log("TensorFlow.js Modelle werden gecacht für DSGVO-konforme Offline-Nutzung");
      },
      (error) => {
        console.log("Service Worker registration failed:", error);
      }
    );
  });
}

createRoot(document.getElementById("root")!).render(<App />);
