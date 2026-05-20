import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [sveltekit()],
  // onnxruntime-web ships large WASM blobs that we don't want bundled.
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  // Required for ORT-Web multithreading (SAB) if we ever enable it.
  // server: {
  //   headers: {
  //     "Cross-Origin-Opener-Policy": "same-origin",
  //     "Cross-Origin-Embedder-Policy": "require-corp",
  //   },
  // },
});
