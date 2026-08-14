import { resolve } from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";


export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
        step01: resolve(import.meta.dirname, "step01/index.html"),
      },
    },
  },
});
