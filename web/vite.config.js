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
        step05: resolve(import.meta.dirname, "step05/index.html"),
        step06: resolve(import.meta.dirname, "step06/index.html"),
        step07: resolve(import.meta.dirname, "step07/index.html"),
        step08: resolve(import.meta.dirname, "step08/index.html"),
        step09: resolve(import.meta.dirname, "step09/index.html"),
        step10: resolve(import.meta.dirname, "step10/index.html"),
        step11: resolve(import.meta.dirname, "step11/index.html"),
        step12: resolve(import.meta.dirname, "step12/index.html"),
      },
    },
  },
});
