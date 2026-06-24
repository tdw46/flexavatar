import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const dirname = path.dirname(fileURLToPath(import.meta.url));
const ortDist = path.join(dirname, "node_modules", "onnxruntime-web", "dist");

function ortWasmFallback() {
  return {
    name: "ort-wasm-fallback",
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const requestPath = request.url?.split("?")[0] ?? "";
        if (!requestPath.includes("ort-wasm") || !requestPath.endsWith(".wasm")) {
          next();
          return;
        }

        const requestedFile = path.basename(requestPath);
        const fallbackFile = requestedFile.includes("jsep")
          ? "ort-wasm-simd-threaded.jsep.wasm"
          : requestedFile.includes("asyncify")
            ? "ort-wasm-simd-threaded.asyncify.wasm"
            : "ort-wasm-simd-threaded.jsep.wasm";
        const wasmPath = path.join(ortDist, fallbackFile);
        if (!fs.existsSync(wasmPath)) {
          next();
          return;
        }

        response.statusCode = 200;
        response.setHeader("Content-Type", "application/wasm");
        response.setHeader("Cache-Control", "no-cache");
        fs.createReadStream(wasmPath).pipe(response);
      });
    },
  };
}

export default defineConfig({
  plugins: [ortWasmFallback(), react()],
  worker: {
    format: "es",
  },
  server: {
    port: 15173,
    proxy: {
      "/api": "http://127.0.0.1:18000",
      "/exports": "http://127.0.0.1:18000",
      "/inputs": "http://127.0.0.1:18000",
    },
  },
});
