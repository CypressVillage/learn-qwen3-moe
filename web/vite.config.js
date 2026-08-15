import { resolve } from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";


export default defineConfig({
  base: process.env.BASE_PATH ?? "/",
  plugins: [react()],
  build: {
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, "index.html"),
        step00: resolve(import.meta.dirname, "step00/index.html"),
        step01: resolve(import.meta.dirname, "step01/index.html"),
        step02: resolve(import.meta.dirname, "step02/index.html"),
        step03: resolve(import.meta.dirname, "step03/index.html"),
        step04: resolve(import.meta.dirname, "step04/index.html"),
      },
    },
  },
});
