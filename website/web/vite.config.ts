import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      // Production Blood GPT inference on Modal (celljepa_s5_s2bjepa_r1).
      "/api": {
        target: "https://ls9563--blood-gpt-bloodgpt-api.us-east.modal.run",
        changeOrigin: true,
        timeout: 180_000,
        proxyTimeout: 180_000,
      },
    },
  },
});
