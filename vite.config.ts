import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig, loadEnv} from 'vite';

export default defineConfig(({mode}) => {
  const env = loadEnv(mode, '.', '');
  const hfToken = env.HUGGINGFACE_TOKEN;
  return {
    plugins: [react(), tailwindcss()],
    define: {
      'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY),
      'process.env.HUGGINGFACE_TOKEN': JSON.stringify(hfToken),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      hmr: process.env.DISABLE_HMR !== 'true',
      proxy: {
        '/api/image': { target: 'http://localhost:3001', changeOrigin: true },
        '/api/chat': { target: 'http://localhost:3001', changeOrigin: true },
        '/api/agent': {
          target: 'http://127.0.0.1:8008',
          changeOrigin: true,
          rewrite: (p) => p.replace(/^\/api\/agent/, ''),
        },
      },
    },
  };
});
