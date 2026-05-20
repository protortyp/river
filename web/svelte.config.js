import adapter from "@sveltejs/adapter-static";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter({
      fallback: "index.html",
    }),
    // Drop-in deploy anywhere static (Cloudflare Pages, Vercel, GitHub Pages).
    paths: { base: "" },
  },
};

export default config;
