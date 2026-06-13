import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:3000', changeOrigin: true }
    }
  },
  preview: {
    // vite preview blocks unknown Host headers; allow the prod domain(s).
    allowedHosts: ['beejhealth.cvframeiq.com']
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: 'terser',
    rollupOptions: { output: { manualChunks: { vendor: ['react','react-dom'] } } }
  }
});
