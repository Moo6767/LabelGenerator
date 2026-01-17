import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { VitePWA } from "vite-plugin-pwa";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8080,
  },
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["favicon.ico", "robots.txt"],
      workbox: {
        globPatterns: ["**/*.{js,css,html,ico,png,svg,woff,woff2}"],
        maximumFileSizeToCacheInBytes: 4 * 1024 * 1024, // 4 MB for TensorFlow.js
        runtimeCaching: [
          {
            // Cache TensorFlow.js models
            urlPattern: /^https:\/\/storage\.googleapis\.com\/tfjs-models\/.*/i,
            handler: "CacheFirst",
            options: {
              cacheName: "tfjs-models-cache",
              expiration: {
                maxEntries: 10,
                maxAgeSeconds: 60 * 60 * 24 * 365, // 1 year
              },
              cacheableResponse: {
                statuses: [0, 200],
              },
            },
          },
          {
            // Cache TensorFlow.js library files
            urlPattern: /^https:\/\/cdn\.jsdelivr\.net\/npm\/@tensorflow.*/i,
            handler: "CacheFirst",
            options: {
              cacheName: "tfjs-lib-cache",
              expiration: {
                maxEntries: 10,
                maxAgeSeconds: 60 * 60 * 24 * 365,
              },
            },
          },
        ],
      },
      manifest: {
        name: "Auto-Annotator",
        short_name: "Annotator",
        description: "Offline KI-Bildannotation - 100% lokal und DSGVO-konform",
        theme_color: "#1a1a1a",
        background_color: "#1a1a1a",
        display: "standalone",
        icons: [
          {
            src: "/favicon.ico",
            sizes: "any",
            type: "image/x-icon",
          },
        ],
      },
    }),
  ].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
