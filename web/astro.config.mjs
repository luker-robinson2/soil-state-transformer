// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import tailwindcss from '@tailwindcss/vite';

// Static site — Tier A is fully precomputed, no backend.
export default defineConfig({
  site: 'https://geosoil.onrender.com',
  integrations: [react()],
  vite: {
    plugins: [tailwindcss()],
  },
});
