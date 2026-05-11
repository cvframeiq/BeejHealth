import express from 'express';
import compression from 'compression';
import helmet from 'helmet';
import cors from 'cors';
import { PORT, FRONTEND } from './config/env.js';
import routes from './routes/index.js';

const app = express();

/* ── Middleware ─────────────────────────────────────────────── */
app.use(compression());
app.use(helmet({ contentSecurityPolicy: false, crossOriginEmbedderPolicy: false }));
app.use(cors({
  origin: [...FRONTEND, 'http://localhost:5173', 'http://localhost:4173', 'http://127.0.0.1:5173'],
  credentials: true,
  methods: ['GET','POST','PUT','PATCH','DELETE','OPTIONS'],
  allowedHeaders: ['Content-Type','Authorization'],
}));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

/* Logger */
app.use((req, _res, next) => {
  if (!req.path.includes('/health'))
    console.log(`[${new Date().toLocaleTimeString()}] ${req.method} ${req.path}`);
  next();
});

/* API Routes */
app.use('/api', routes);

/* 404 */
app.use('/api/*', (req, res) => res.status(404).json({ error: `Route nahi mila: ${req.path}` }));

/* Error handler */
app.use((err, _req, res, next) => {
  if (!err) return next();
  if (err.type === 'entity.too.large') {
    return res.status(413).json({ error: 'Image payload bahut badi hai. Thodi chhoti ya compressed photo upload karein.' });
  }
  if (err instanceof SyntaxError && 'body' in err) {
    return res.status(400).json({ error: 'Upload data sahi format mein nahi aayi' });
  }
  console.error('Unhandled:', err);
  return res.status(500).json({ error: 'Internal server error' });
});

/* ── Start ── */
app.listen(PORT, () => {
  console.log('\n🌱 ════════════════════════════════════════');
  console.log('   BeejHealth API v3.0 chal raha hai! (MVC Refactored)');
  console.log(`   🚀  http://localhost:${PORT}/api/health`);
  console.log(`   📊  http://localhost:${PORT}/api/stats`);
  console.log(`   👥  http://localhost:${PORT}/api/admin/users`);
  console.log('🌱 ════════════════════════════════════════\n');
});
