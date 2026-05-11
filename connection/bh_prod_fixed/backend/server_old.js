// import express    from 'express';
// import compression from 'compression';
// import helmet      from 'helmet';
// import cors        from 'cors';
// import bcrypt      from 'bcryptjs';
// import jwt         from 'jsonwebtoken';
// import path        from 'path';
// import { readFileSync } from 'fs';
// import { fileURLToPath } from 'url';
// import {
//   usersDB, consultationsDB, notificationsDB, chatsDB,
//   robotsDB, robotLogsDB, sprayJobsDB, photosDB,
//   dbFind, dbFindOne, dbInsert, dbUpdate, dbRemove, dbCount,
// } from './db.js';

// const __dirname  = path.dirname(fileURLToPath(import.meta.url));
// const PORT       = process.env.PORT       || 3000;
// const JWT_SECRET = process.env.JWT_SECRET || 'bh-jwt-secret-change-in-production-2024';
// const FRONTEND   = (process.env.FRONTEND_URL || 'http://localhost:5173').split(',');
// const AI_SERVER  = process.env.AI_SERVER_URL || 'http://localhost:8000'; // Python FastAPI: ai_service/ai_server.py

// const app = express();

// /* ── Middleware ─────────────────────────────────────────────── */
// app.use(compression());
// app.use(helmet({ contentSecurityPolicy: false, crossOriginEmbedderPolicy: false }));
// app.use(cors({
//   origin: [...FRONTEND, 'http://localhost:5173', 'http://localhost:4173', 'http://127.0.0.1:5173'],
//   credentials: true,
//   methods: ['GET','POST','PUT','PATCH','DELETE','OPTIONS'],
//   allowedHeaders: ['Content-Type','Authorization'],
// }));
// app.use(express.json({ limit: '35mb' }));
// app.use(express.urlencoded({ extended: true, limit: '35mb' }));

// /* ── Photo Upload — Multi-photo, DB-backed ───────────────────── */
// import { randomBytes } from 'crypto';
// const uploads = new Map(); // in-memory cache for fast retrieval
// const MAX_PHOTO_UPLOAD_BYTES = 25 * 1024 * 1024;
// const USERS_DB_FILE = path.join(__dirname, 'data', 'users.db');

// const normalizeMobile = (mobile) => String(mobile ?? '').replace(/\D/g, '').slice(-10);

// function readUsersFromDisk() {
//   try {
//     const raw = readFileSync(USERS_DB_FILE, 'utf8');
//     return raw
//       .split('\n')
//       .map((line) => {
//         if (!line || line.includes('"$$deleted"') || line.includes('"$$indexCreated"')) return null;
//         try { return JSON.parse(line); } catch { return null; }
//       })
//       .filter(Boolean);
//   } catch (e) {
//     console.warn('users.db fallback read failed:', e.message);
//     return [];
//   }
// }

// function pickLatestUser(users) {
//   return [...users].sort((a, b) => {
//     const aTime = new Date(a.lastLogin || a.createdAt || 0).getTime();
//     const bTime = new Date(b.lastLogin || b.createdAt || 0).getTime();
//     if (bTime !== aTime) return bTime - aTime;
//     return String(b._id || '').localeCompare(String(a._id || ''));
//   })[0] || null;
// }

// function uniqUsers(users) {
//   const seen = new Set();
//   return users.filter((user) => {
//     if (!user) return false;
//     const key = String(
//       user._id ||
//       `${normalizeMobile(user.mobile)}|${user.type || ''}|${user.createdAt || ''}|${user.lastLogin || ''}`
//     );
//     if (seen.has(key)) return false;
//     seen.add(key);
//     return true;
//   });
// }

// function sortUsersForLogin(users, preferredType = null) {
//   return [...users].sort((a, b) => {
//     if (preferredType) {
//       const aPref = a.type === preferredType;
//       const bPref = b.type === preferredType;
//       if (aPref !== bPref) return aPref ? -1 : 1;
//     }
//     const aTime = new Date(a.lastLogin || a.createdAt || 0).getTime();
//     const bTime = new Date(b.lastLogin || b.createdAt || 0).getTime();
//     if (bTime !== aTime) return bTime - aTime;
//     return String(b._id || '').localeCompare(String(a._id || ''));
//   });
// }

// async function findUsersByMobile(rawMobile, preferredType = null) {
//   const mobile = normalizeMobile(rawMobile);
//   if (mobile.length < 10) return [];

//   const diskUsers = readUsersFromDisk().filter((u) => normalizeMobile(u.mobile) === mobile);
//   let liveUsers = [];
//   try {
//     liveUsers = await dbFind(usersDB, { mobile });
//   } catch (e) {
//     console.warn('users lookup failed:', e.message);
//   }

//   return sortUsersForLogin(uniqUsers([...diskUsers, ...liveUsers]), preferredType);
// }

// async function findUserByMobile(rawMobile) {
//   const users = await findUsersByMobile(rawMobile);
//   return users[0] || null;
// }
// const MIN_REPORT_PHOTOS = 3;
// const MAX_REPORT_PHOTOS = 5;

// async function linkPhotosToConsultation(photoIds, consultationId, userId) {
//   const ids = Array.isArray(photoIds) ? photoIds.filter(Boolean) : [];
//   if (!consultationId || ids.length === 0) return;
//   await Promise.all(ids.map((photoId) =>
//     dbUpdate(photosDB,
//       { photoId, userId },
//       { $set: { consultationId } }
//     ).catch(() => {})
//   ));
// }

// /* POST /api/photos — upload 1 photo (call multiple times for multiple photos) */
// app.post('/api/photos', requireAuth, async (req, res) => {
//   try {
//     const { base64, type = 'image/jpeg', consultationId, index = 0, totalExpected = 1 } = req.body;
//     if (!base64) return res.status(400).json({ error: 'Image data missing' });
//     if (!base64.startsWith('data:image'))
//       return res.status(400).json({ error: 'Valid image chahiye (JPG/PNG/HEIC)' });

//     // Size check: 25MB limit per photo before base64/JSON expansion
//     const sizeBytes = Math.round((base64.length * 3) / 4);
//     if (sizeBytes > MAX_PHOTO_UPLOAD_BYTES)
//       return res.status(400).json({ error: 'Ek photo max 25MB honi chahiye' });

//     // Max 5 photos per consultation
//     if (consultationId) {
//       const existing = await dbFind(photosDB, { consultationId, userId: req.user.id });
//       if (existing.length >= MAX_REPORT_PHOTOS)
//         return res.status(400).json({ error: `Maximum ${MAX_REPORT_PHOTOS} photos allowed per consultation` });
//     }

//     const photoId = randomBytes(8).toString('hex');

//     // Save to DB (metadata + base64)
//     const photoDoc = await dbInsert(photosDB, {
//       photoId,
//       userId:         req.user.id,
//       consultationId: consultationId || null,
//       type,
//       base64,          // stored in DB
//       index:           Number(index),
//       totalExpected:   Number(totalExpected),
//       sizeKB:          Math.round(sizeBytes / 1024),
//       url:             `/api/photos/${photoId}`,
//       createdAt:       ts(),
//     });

//     // Also cache in memory for fast serving
//     uploads.set(photoId, { base64, type, userId: req.user.id });

//     // If consultationId, update consultation with this photo
//     if (consultationId) {
//       const existing = await dbFind(photosDB, { consultationId, userId: req.user.id });
//       const photoIds  = existing.map(p => p.photoId);
//       const photoUrls = existing.map(p => p.url);
//       await dbUpdate(consultationsDB,
//         { _id: consultationId },
//         { $set: {
//           photoUploaded: true,
//           photoId:   photoIds[0],
//           photoUrl:  photoUrls[0],
//           photoIds,
//           photoUrls,
//           photoCount: existing.length,
//           updatedAt: ts(),
//         }}
//       ).catch(() => {});
//     }

//     console.log(`📸 Photo ${Number(index)+1}/${totalExpected} uploaded: ${photoId} (${Math.round(sizeBytes/1024)}KB) user:${req.user.id}`);
//     res.json({ success: true, photoId, url: `/api/photos/${photoId}`, sizeKB: Math.round(sizeBytes / 1024), index });
//   } catch (e) { console.error('photo upload:', e); res.status(500).json({ error: 'Upload fail hua' }); }
// });

// /* Legacy endpoint — keep working */
// app.post('/api/upload/photo', requireAuth, async (req, res) => {
//   req.body.index = 0; req.body.totalExpected = 1;
//   const { base64, type = 'image/jpeg', consultationId } = req.body;
//   if (!base64) return res.status(400).json({ error: 'Image data missing' });
//   const photoId = randomBytes(8).toString('hex');
//   uploads.set(photoId, { base64, type, userId: req.user.id });
//   await dbInsert(photosDB, { photoId, userId: req.user.id, consultationId: consultationId||null, type, base64, index:0, sizeKB: Math.round((base64.length*3/4)/1024), url:`/api/photos/${photoId}`, createdAt: ts() }).catch(()=>{});
//   if (consultationId) await dbUpdate(consultationsDB, { _id: consultationId }, { $set: { photoId, photoUrl:`/api/photos/${photoId}`, photoUploaded:true }}).catch(()=>{});
//   console.log(`📸 (legacy) Photo: ${photoId} user:${req.user.id}`);
//   res.json({ success: true, photoId, url: `/api/photos/${photoId}` });
// });

// /* GET /api/photos/:photoId — serve a photo */
// app.get('/api/photos/:photoId', async (req, res) => {
//   try {
//     // Check memory cache first (fast)
//     let photo = uploads.get(req.params.photoId);
//     if (!photo) {
//       // Fall back to DB
//       const doc = await dbFindOne(photosDB, { photoId: req.params.photoId });
//       if (!doc) return res.status(404).json({ error: 'Photo nahi mili' });
//       photo = doc;
//       uploads.set(req.params.photoId, photo); // re-cache
//     }
//     const rawBase64 = photo.base64.includes(',') ? photo.base64.split(',')[1] : photo.base64;
//     const buffer = Buffer.from(rawBase64, 'base64');
//     res.set('Content-Type', photo.type || 'image/jpeg');
//     res.set('Cache-Control', 'public, max-age=3600');
//     res.send(buffer);
//   } catch (e) { res.status(500).json({ error: 'Photo serve fail' }); }
// });

// /* Legacy GET */
// app.get('/api/upload/photo/:id', async (req, res) => {
//   req.params.photoId = req.params.id;
//   const photo = uploads.get(req.params.id) || await dbFindOne(photosDB, { photoId: req.params.id }).catch(()=>null);
//   if (!photo) return res.status(404).json({ error: 'Photo nahi mili' });
//   const buf = Buffer.from((photo.base64||'').split(',')[1]||(photo.base64||''), 'base64');
//   res.set('Content-Type', photo.type||'image/jpeg');
//   res.send(buf);
// });

// /* GET /api/photos — list photos for a consultation */
// app.get('/api/photos', requireAuth, async (req, res) => {
//   try {
//     const { consultationId } = req.query;
//     const q = { userId: req.user.id };
//     if (consultationId) q.consultationId = consultationId;
//     const photos = await dbFind(photosDB, q);
//     photos.sort((a, b) => a.index - b.index);
//     res.json({
//       photos: photos.map(p => ({
//         photoId: p.photoId, url: p.url, index: p.index,
//         sizeKB: p.sizeKB, type: p.type, createdAt: p.createdAt,
//         consultationId: p.consultationId,
//       })),
//       count: photos.length,
//     });
//   } catch (e) { res.status(500).json({ error: 'Photos nahi mile' }); }
// });

// /* DELETE /api/photos/:photoId — delete a photo */
// app.delete('/api/photos/:photoId', requireAuth, async (req, res) => {
//   try {
//     const photo = await dbFindOne(photosDB, { photoId: req.params.photoId, userId: req.user.id });
//     if (!photo) return res.status(404).json({ error: 'Photo nahi mili ya aapki nahi hai' });
//     await dbRemove(photosDB, { photoId: req.params.photoId });
//     uploads.delete(req.params.photoId);
//     // Update consultation photo count
//     if (photo.consultationId) {
//       const remaining = await dbFind(photosDB, { consultationId: photo.consultationId });
//       await dbUpdate(consultationsDB, { _id: photo.consultationId },
//         { $set: { photoCount: remaining.length, photoUploaded: remaining.length > 0 }}
//       ).catch(() => {});
//     }
//     res.json({ success: true, message: 'Photo delete ho gayi' });
//   } catch (e) { res.status(500).json({ error: 'Delete fail' }); }
// });

// /* ── AI Server Helper ───────────────────────────────────────────── */
// async function callAI(endpoint, body) {
//   try {
//     const res = await fetch(`${AI_SERVER}${endpoint}`, {
//       method: 'POST',
//       headers: { 'Content-Type': 'application/json' },
//       body: JSON.stringify(body),
//       signal: AbortSignal.timeout(30000),
//     });
//     if (!res.ok) {
//       const err = await res.json().catch(() => ({ error: 'AI server error' }));
//       throw new Error(err.detail || err.error || `AI ${res.status}`);
//     }
//     return res.json();
//   } catch(e) {
//     if (e.code === 'ECONNREFUSED' || e.name === 'TimeoutError') {
//       throw new Error('AI_SERVER_DOWN');
//     }
//     throw e;
//   }
// }

// /* Logger */
// app.use((req, _res, next) => {
//   if (!req.path.includes('/health'))
//     console.log(`[${new Date().toLocaleTimeString()}] ${req.method} ${req.path}`);
//   next();
// });

// /* ── Helpers ─────────────────────────────────────────────────  */
// const ts   = () => new Date().toISOString();
// const safe = (u) => { if (!u) return null; const { password: _, ...r } = u; return r; };
// const sign = (u) => jwt.sign({ id: u._id, mobile: u.mobile, type: u.type }, JWT_SECRET, { expiresIn: '30d' });

// /* ── Auth Guards ─────────────────────────────────────────────  */
// function requireAuth(req, res, next) {
//   const token = req.headers.authorization?.split(' ')[1];
//   if (!token) return res.status(401).json({ error: 'Token nahi mila. Login karein.' });
//   try { req.user = jwt.verify(token, JWT_SECRET); next(); }
//   catch { res.status(401).json({ error: 'Session expire ho gaya. Dobara login karein.' }); }
// }
// function requireExpert(req, res, next) {
//   if (req.user?.type !== 'expert') return res.status(403).json({ error: 'Sirf experts ke liye.' });
//   next();
// }

// /* ═══════════════════════════════════════════════════════════════
//    AUTH
// ═══════════════════════════════════════════════════════════════ */

// /* POST /api/auth/send-otp */
// app.post('/api/auth/send-otp', async (req, res) => {
//   try {
//     const { mobile } = req.body;
//     const cleanMobile = normalizeMobile(mobile);
//     if (cleanMobile.length < 10)
//       return res.status(400).json({ error: 'Valid 10-digit mobile number daalein' });
//     console.log(`📱 OTP [demo] → ${cleanMobile} : 123456`);
//     res.json({ success: true, message: 'OTP bheja gaya! (Demo mode — OTP hai: 123456)' });
//   } catch (e) { res.status(500).json({ error: 'Server error' }); }
// });

// /* POST /api/auth/login */
// app.post('/api/auth/login', async (req, res) => {
//   try {
//     const { mobile, password, otp, method, type } = req.body;
//     const cleanMobile = normalizeMobile(mobile);
//     if (cleanMobile.length < 10) return res.status(400).json({ error: 'Mobile number daalein' });

//     const candidates = await findUsersByMobile(cleanMobile, type);
//     if (!candidates.length) return res.status(404).json({ error: 'Yeh mobile registered nahi. Pehle register karein.' });

//     let user = null;
//     if (method === 'otp') {
//       if (!otp || String(otp).length < 6)
//         return res.status(400).json({ error: '6-digit OTP enter karein' });
//       /* Demo: any 6-digit OTP accepted */
//       user = candidates[0];
//     } else {
//       if (!password) return res.status(400).json({ error: 'Password enter karein' });
//       for (const candidate of candidates) {
//         if (!candidate?.password) continue;
//         const ok = await bcrypt.compare(String(password), candidate.password);
//         if (ok) {
//           user = candidate;
//           break;
//         }
//       }
//       if (!user) return res.status(401).json({ error: 'Password galat hai. Dobara try karein.' });
//     }

//     await dbUpdate(usersDB, { _id: user._id }, { $set: { lastLogin: ts() } });
//     console.log(`✅ Login: ${user.name} (${user.type})`);
//     res.json({ success: true, token: sign(user), user: safe(user) });
//   } catch (e) { console.error('login:', e); res.status(500).json({ error: 'Server error. Dobara try karein.' }); }
// });

// /* POST /api/auth/register */
// app.post('/api/auth/register', async (req, res) => {
//   try {
//     const { name, mobile, email, password, type,
//             district, taluka, village, soil,
//             spec, fee, university, langs, crops } = req.body;

//     /* Validation */
//     if (!name     || name.trim().length < 2)     return res.status(400).json({ error: 'Naam 2+ characters hona chahiye' });
//     const cleanMobile = normalizeMobile(mobile);
//     if (cleanMobile.length < 10)                 return res.status(400).json({ error: 'Valid 10-digit mobile daalein' });
//     if (!password || password.length < 8)         return res.status(400).json({ error: 'Password 8+ characters hona chahiye' });
//     if (type === 'expert' && !spec)               return res.status(400).json({ error: 'Specialization daalein' });

//     const exists = await findUserByMobile(cleanMobile);
//     if (exists) return res.status(409).json({ error: 'Yeh mobile pehle se registered hai. Login karein.' });

//     const trimName = name.trim();
//     const initials = trimName.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
//     const hashed   = await bcrypt.hash(password, 12);

//     const user = await dbInsert(usersDB, {
//       name:       trimName,
//       mobile:     cleanMobile,
//       email:      email?.trim() || '',
//       password:   hashed,
//       type:       type || 'farmer',
//       initials,
//       state:      req.body.state      || 'Maharashtra',
//       district:   district            || '',
//       taluka:     req.body.taluka     || '',
//       village:    village             || '',
//       soil:       soil                || '',
//       farmSize:   Number(req.body.farmSize) || 0,
//       irrigation: req.body.irrigation || '',
//       spec:       spec                || '',
//       fee:        Number(fee)         || 0,
//       university: university          || '',
//       langs:      langs               || 'Hindi',
//       expYrs:     Number(req.body.expYrs) || 0,
//       crops:      Array.isArray(crops) ? crops : [],
//       verified:   false,
//       available:  type === 'expert',
//       rating:     0,
//       totalCases: 0,
//       bio:        '',
//       createdAt:  ts(),
//       lastLogin:  ts(),
//     });

//     /* Welcome notification */
//     await dbInsert(notificationsDB, {
//       userId:    user._id,
//       type:      'welcome',
//       icon:      '🌱',
//       title:     'BeejHealth mein Swagat! 🎉',
//       body:      `Namaste ${trimName}! Aapka account ban gaya hai. ${type === 'expert' ? 'Ab aap cases receive kar sakte hain.' : 'Ab aap crop consultation le sakte hain.'}`,
//       read:      false,
//       createdAt: ts(),
//     });

//     console.log(`🆕 Register: ${trimName} (${user.type})`);
//     res.json({ success: true, token: sign(user), user: { ...safe(user), fresh: true } });
//   } catch (e) {
//     if (e.errorType === 'uniqueViolated') return res.status(409).json({ error: 'Mobile already registered' });
//     console.error('register:', e);
//     res.status(500).json({ error: 'Registration fail hua. Console check karein.' });
//   }
// });

// /* GET /api/auth/me */
// app.get('/api/auth/me', requireAuth, async (req, res) => {
//   try {
//     const user = await dbFindOne(usersDB, { _id: req.user.id });
//     if (!user) return res.status(404).json({ error: 'User nahi mila' });
//     res.json({ user: safe(user) });
//   } catch (e) { res.status(500).json({ error: 'Server error' }); }
// });

// /* PATCH /api/auth/profile */
// app.patch('/api/auth/profile', requireAuth, async (req, res) => {
//   try {
//     const allowed = ['name','email','state','district','taluka','village','soil',
//                      'farmSize','irrigation','spec','fee','langs','crops','available','university','bio','expYrs'];
//     const upd = {};
//     allowed.forEach(k => { if (req.body[k] !== undefined) upd[k] = req.body[k]; });
//     if (upd.name) upd.initials = upd.name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
//     await dbUpdate(usersDB, { _id: req.user.id }, { $set: upd });
//     const user = await dbFindOne(usersDB, { _id: req.user.id });
//     res.json({ success: true, user: safe(user) });
//   } catch (e) { res.status(500).json({ error: 'Profile update fail' }); }
// });

// /* PATCH /api/auth/password */
// app.patch('/api/auth/password', requireAuth, async (req, res) => {
//   try {
//     const { oldPassword, newPassword } = req.body;
//     if (!newPassword || newPassword.length < 8)
//       return res.status(400).json({ error: 'Naya password 8+ characters' });
//     const user = await dbFindOne(usersDB, { _id: req.user.id });
//     const ok   = await bcrypt.compare(oldPassword, user.password);
//     if (!ok) return res.status(401).json({ error: 'Purana password galat hai' });
//     const hashed = await bcrypt.hash(newPassword, 12);
//     await dbUpdate(usersDB, { _id: req.user.id }, { $set: { password: hashed } });
//     res.json({ success: true, message: 'Password successfully change ho gaya!' });
//   } catch (e) { res.status(500).json({ error: 'Password update fail' }); }
// });

// /* ═══════════════════════════════════════════════════════════════
//    EXPERTS
// ═══════════════════════════════════════════════════════════════ */

// /* GET /api/experts */
// app.get('/api/experts', async (req, res) => {
//   try {
//     const q = { type: 'expert' };
//     if (req.query.spec)             q.spec      = req.query.spec;
//     if (req.query.available === 'true') q.available = true;
//     const list = await dbFind(usersDB, q);
//     res.json({ experts: list.map(safe) });
//   } catch (e) { res.status(500).json({ error: 'Experts nahi mile' }); }
// });

// /* GET /api/experts/:id */
// app.get('/api/experts/:id', async (req, res) => {
//   try {
//     const expert = await dbFindOne(usersDB, { _id: req.params.id, type: 'expert' });
//     if (!expert) return res.status(404).json({ error: 'Expert nahi mila' });
//     res.json({ expert: safe(expert) });
//   } catch (e) { res.status(500).json({ error: 'Server error' }); }
// });

// /* PATCH /api/experts/availability */
// app.patch('/api/experts/availability', requireAuth, requireExpert, async (req, res) => {
//   try {
//     const available = !!req.body.available;
//     await dbUpdate(usersDB, { _id: req.user.id }, { $set: { available } });
//     console.log(`👨‍⚕️ Expert ${req.user.id}: available=${available}`);
//     res.json({ success: true, available });
//   } catch (e) { res.status(500).json({ error: 'Availability update fail' }); }
// });

// /* ═══════════════════════════════════════════════════════════════
//    CONSULTATIONS
// ═══════════════════════════════════════════════════════════════ */

// /* POST /api/consultations */
// app.post('/api/consultations', requireAuth, async (req, res) => {
//   try {
//     const { cropId, cropName, cropEmoji, method, photoUploaded,
//             answers, disease, confidence, severity } = req.body;
//     const photoId = req.body.photoId || null;
//     const photoUrl = req.body.photoUrl || null;
//     const photoIds = Array.isArray(req.body.photoIds) ? req.body.photoIds.filter(Boolean) : (photoId ? [photoId] : []);
//     const photoUrls = Array.isArray(req.body.photoUrls) ? req.body.photoUrls.filter(Boolean) : (photoUrl ? [photoUrl] : []);
//     const resolvedPhotoCount = Math.max(
//       Number(req.body.photoCount) || 0,
//       photoIds.length,
//       photoUrls.length,
//       photoId ? 1 : 0,
//     );
//     const hasPhotoSet = resolvedPhotoCount > 0 || photoIds.length > 0 || photoUrls.length > 0 || !!photoUploaded || method === 'photo';
//     if (resolvedPhotoCount > MAX_REPORT_PHOTOS)
//       return res.status(400).json({ error: `Maximum ${MAX_REPORT_PHOTOS} photos allowed per consultation` });
//     if (hasPhotoSet && resolvedPhotoCount < MIN_REPORT_PHOTOS)
//       return res.status(400).json({ error: `Minimum ${MIN_REPORT_PHOTOS} photos required for report generation` });

//     /* Auto-assign a random available expert */
//     const experts = await dbFind(usersDB, { type: 'expert', available: true });
//     const expert  = experts.length > 0
//       ? experts[Math.floor(Math.random() * experts.length)] : null;

//     const c = await dbInsert(consultationsDB, {
//       farmerId:      req.user.id,
//       expertId:      expert?._id   || null,
//       expertName:    expert?.name  || 'Pending Assignment',
//       cropId:        cropId        || 'tomato',
//       cropName:      cropName      || 'Crop',
//       cropEmoji:     cropEmoji     || '🌱',
//       method:        method        || 'photo',
//       photoUploaded: resolvedPhotoCount >= MIN_REPORT_PHOTOS,
//       photoId:       photoIds[0]    || photoId || null,
//       photoUrl:      photoUrls[0]   || photoUrl || null,
//       photoIds,
//       photoUrls,
//       photoCount:    resolvedPhotoCount,
//       answers:       answers       || {},
//       disease:       disease       || 'Analysis Pending',
//       confidence:    Number(confidence) || 0,
//       severity:      Number(severity)   || 1,
//       status:        expert ? 'expert_assigned' : 'pending',
//       report:        null,
//       createdAt:     ts(),
//       updatedAt:     ts(),
//     });

//     await linkPhotosToConsultation(photoIds, c._id, req.user.id);

//     /* Notify farmer */
//     await dbInsert(notificationsDB, {
//       userId: req.user.id, type: 'consultation', icon: '🔬',
//       title:  `AI Report Ready — ${cropName}`,
//       body:   `${disease} detected (${confidence}% confidence).${expert ? ` Dr. ${expert.name} assign ho gaya.` : ''}`,
//       read:   false, consultationId: c._id, createdAt: ts(),
//     });

//     /* Notify expert */
//     if (expert) {
//       await dbInsert(notificationsDB, {
//         userId: expert._id, type: 'new_case', icon: '📋',
//         title:  `Naya Case — ${cropName}`,
//         body:   `${disease} suspected. Immediately review karein.`,
//         read:   false, consultationId: c._id, createdAt: ts(),
//       });
//       await dbUpdate(usersDB, { _id: expert._id }, { $inc: { totalCases: 1 } });
//     }

//     console.log(`📋 New consultation: ${cropName} | ${disease} | expert: ${expert?.name || 'none'}`);
//     res.json({ success: true, consultation: c });
//   } catch (e) { console.error('consultation:', e); res.status(500).json({ error: 'Consultation save nahi hua' }); }
// });

// /* GET /api/consultations */
// app.get('/api/consultations', requireAuth, async (req, res) => {
//   try {
//     const q = req.user.type === 'expert'
//       ? { expertId: req.user.id }
//       : { farmerId: req.user.id };
//     const list = await dbFind(consultationsDB, q);
//     list.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
//     res.json({ consultations: list });
//   } catch (e) { res.status(500).json({ error: 'Consultations nahi mile' }); }
// });

// /* GET /api/consultations/:id */
// app.get('/api/consultations/:id', requireAuth, async (req, res) => {
//   try {
//     const c = await dbFindOne(consultationsDB, { _id: req.params.id });
//     if (!c) return res.status(404).json({ error: 'Consultation nahi mili' });
//     res.json({ consultation: c });
//   } catch (e) { res.status(500).json({ error: 'Server error' }); }
// });

// /* PATCH /api/consultations/:id/status */
// app.patch('/api/consultations/:id/status', requireAuth, async (req, res) => {
//   try {
//     const { status, report } = req.body;
//     const upd = { status, updatedAt: ts() };
//     if (report) upd.report = report;
//     await dbUpdate(consultationsDB, { _id: req.params.id }, { $set: upd });

//     if (status === 'completed') {
//       const c = await dbFindOne(consultationsDB, { _id: req.params.id });
//       if (c) {
//         const expertUser = await dbFindOne(usersDB, { _id: req.user.id });
//         await dbInsert(notificationsDB, {
//           userId: c.farmerId, type: 'report_ready', icon: '✅',
//           title:  `Expert Report Ready — ${c.cropName}!`,
//           body:   `Dr. ${expertUser?.name || 'Expert'} ne aapki ${c.cropName} ki full report bhej di. Abhi dekhein!`,
//           read:   false, consultationId: c._id, createdAt: ts(),
//         });
//       }
//     }
//     res.json({ success: true });
//   } catch (e) { res.status(500).json({ error: 'Status update fail' }); }
// });

// /* PATCH /api/consultations/:id/report */
// app.patch('/api/consultations/:id/report', requireAuth, async (req, res) => {
//   try {
//     const c = await dbFindOne(consultationsDB, { _id: req.params.id });
//     if (!c) return res.status(404).json({ error: 'Consultation nahi mili' });
//     if (req.user.id !== c.farmerId && req.user.id !== c.expertId)
//       return res.status(403).json({ error: 'Aapko is report par access nahi hai' });

//     const { reportSnapshot, reportSummary, report } = req.body;
//     if (!reportSnapshot && !reportSummary && !report)
//       return res.status(400).json({ error: 'Report data missing' });

//     const upd = { updatedAt: ts() };
//     if (reportSnapshot) {
//       upd.reportSnapshot = { ...(c.reportSnapshot || {}), ...reportSnapshot };
//       if (reportSnapshot.generatedAt) upd.reportGeneratedAt = reportSnapshot.generatedAt;
//       if (reportSnapshot.downloadedAt) upd.reportDownloadedAt = reportSnapshot.downloadedAt;
//     }
//     if (reportSummary || report) {
//       const summaryText = reportSummary || report;
//       upd.aiReportSummary = summaryText;
//       if (!c.report) upd.report = summaryText;
//     }

//     await dbUpdate(consultationsDB, { _id: req.params.id }, { $set: upd });
//     const saved = await dbFindOne(consultationsDB, { _id: req.params.id });
//     res.json({ success: true, consultation: saved });
//   } catch (e) {
//     console.error('report save:', e);
//     res.status(500).json({ error: 'Report save nahi hua' });
//   }
// });

// /* ═══════════════════════════════════════════════════════════════
//    CHAT
// ═══════════════════════════════════════════════════════════════ */

// /* GET /api/consultations/:id/messages */
// app.get('/api/consultations/:id/messages', requireAuth, async (req, res) => {
//   try {
//     const msgs = await dbFind(chatsDB, { consultationId: req.params.id });
//     msgs.sort((a, b) => new Date(a.createdAt) - new Date(b.createdAt));
//     res.json({ messages: msgs });
//   } catch (e) { res.status(500).json({ error: 'Messages nahi mile' }); }
// });

// /* POST /api/consultations/:id/messages */
// app.post('/api/consultations/:id/messages', requireAuth, async (req, res) => {
//   try {
//     const { text } = req.body;
//     if (!text?.trim()) return res.status(400).json({ error: 'Message empty nahi ho sakta' });

//     const sender = await dbFindOne(usersDB, { _id: req.user.id });
//     const msg    = await dbInsert(chatsDB, {
//       consultationId: req.params.id,
//       senderId:   req.user.id,
//       senderName: sender?.name || 'User',
//       senderType: req.user.type,
//       text:       text.trim(),
//       createdAt:  ts(),
//     });

//     /* Notify other party */
//     const c    = await dbFindOne(consultationsDB, { _id: req.params.id });
//     const toId = c && (req.user.type === 'expert' ? c.farmerId : c.expertId);
//     if (toId) {
//       await dbInsert(notificationsDB, {
//         userId: toId, type: 'message', icon: '💬',
//         title:  `${sender?.name} ne message bheja`,
//         body:   text.trim().slice(0, 100),
//         read:   false, consultationId: req.params.id, createdAt: ts(),
//       });
//     }
//     res.json({ success: true, message: msg });
//   } catch (e) { res.status(500).json({ error: 'Message send nahi hua' }); }
// });

// /* ═══════════════════════════════════════════════════════════════
//    NOTIFICATIONS
// ═══════════════════════════════════════════════════════════════ */

// app.get('/api/notifications', requireAuth, async (req, res) => {
//   try {
//     const list = await dbFind(notificationsDB, { userId: req.user.id });
//     list.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
//     res.json({ notifications: list });
//   } catch (e) { res.status(500).json({ error: 'Notifications nahi mile' }); }
// });

// app.patch('/api/notifications/read-all', requireAuth, async (req, res) => {
//   await dbUpdate(notificationsDB, { userId: req.user.id, read: false },
//     { $set: { read: true } }, { multi: true });
//   res.json({ success: true });
// });

// app.patch('/api/notifications/:id/read', requireAuth, async (req, res) => {
//   await dbUpdate(notificationsDB, { _id: req.params.id, userId: req.user.id },
//     { $set: { read: true } });
//   res.json({ success: true });
// });

// /* ═══════════════════════════════════════════════════════════════
//    STATS & ADMIN
// ═══════════════════════════════════════════════════════════════ */

// /* GET /api/crops/coconut/questions — Coconut adaptive questions bank */
// app.get('/api/crops/coconut/questions', (_, res) => {
//   res.json({
//     crop: 'coconut',
//     model: 'EfficientNetV2-S (Transfer Learning)',
//     diseases: [
//       { id:'gray_leaf_spot',    name:'Gray Leaf Spot',    sci:'Pestalotiopsis palmarum',    sev:2 },
//       { id:'gray_leaf_spot_m', name:'Gray Leaf Spot (Multiple)', sci:'Pestalotiopsis palmarum', sev:3 },
//       { id:'leaf_rot',         name:'Leaf Rot',           sci:'Colletotrichum gloeosporioides', sev:3 },
//       { id:'leaf_rot_m',      name:'Leaf Rot (Multiple)', sci:'Colletotrichum gloeosporioides', sev:4 },
//       { id:'bud_rot',          name:'Bud Rot',            sci:'Phytophthora palmivora',     sev:5 },
//       { id:'bud_rot_m',       name:'Bud Rot (Multiple)', sci:'Phytophthora palmivora',     sev:5 },
//       { id:'stem_bleeding',   name:'Stem Bleeding',       sci:'Thielaviopsis paradoxa',    sev:4 },
//       { id:'stem_bleeding_m', name:'Stem Bleeding (Multiple)', sci:'Thielaviopsis paradoxa', sev:4 },
//       { id:'healthy',         name:'Healthy',             sci:'—',                          sev:0 },
//     ],
//     total_questions: 30,
//     questions_per_session: 5,
//     branches: {
//       coconut_spots:   { desc:'Patte pe daag — Gray Leaf Spot ya Leaf Rot', q_count: 5 },
//       coconut_wilt:    { desc:'Stem Bleeding ya Bud Rot symptoms',          q_count: 5 },
//       coconut_yellow:  { desc:'Yellowing — nutrient ya disease',            q_count: 5 },
//       coconut_none:    { desc:'Preventive check — healthy tree',            q_count: 5 },
//       coconut_advanced:{ desc:'Advanced diagnostics (any branch)',          q_count: 9 },
//     }
//   });
// });

// /* POST /api/experts/:id/rate — Rate an expert after consultation */
// app.post('/api/experts/:id/rate', requireAuth, async (req, res) => {
//   try {
//     const { rating, consultationId } = req.body;
//     if(!rating || rating < 1 || rating > 5)
//       return res.status(400).json({ error: 'Rating 1-5 ke beech honi chahiye' });

//     const expert = await dbFindOne(usersDB, { _id: req.params.id, type: 'expert' });
//     if(!expert) return res.status(404).json({ error: 'Expert nahi mila' });

//     // Weighted average: (oldRating * totalCases + newRating) / (totalCases + 1)
//     const oldRating   = expert.rating || 4.5;
//     const totalCases  = expert.totalCases || 1;
//     const newRating   = ((oldRating * totalCases) + rating) / (totalCases + 1);

//     await dbUpdate(usersDB, { _id: req.params.id }, {
//       $set: { rating: Math.round(newRating * 10) / 10 }
//     });

//     // Notify expert
//     await dbInsert(notificationsDB, {
//       userId: req.params.id, type: 'rating', icon: '⭐',
//       title: `Naya Rating Mila — ${rating}/5`,
//       body: `Ek farmer ne aapko ${rating} star diye consultation ke baad.`,
//       read: false, createdAt: ts(),
//     });

//     console.log(`⭐ Rating: expert ${req.params.id} → ${newRating.toFixed(1)}`);
//     res.json({ success: true, newRating: Math.round(newRating * 10) / 10 });
//   } catch(e) { console.error('rate:', e); res.status(500).json({ error: 'Rating save nahi hua' }); }
// });

// /* GET /api/earnings — Expert earnings summary */
// app.get('/api/earnings', requireAuth, requireExpert, async (req, res) => {
//   try {
//     const consults = await dbFind(consultationsDB, { expertId: req.user.id });
//     const expert   = await dbFindOne(usersDB, { _id: req.user.id });
//     const fee      = expert?.fee || 500;
//     const completed = consults.filter(c => c.status === 'completed');
//     const now = new Date();
//     const thisMonth = completed.filter(c => {
//       const d = new Date(c.createdAt);
//       return d.getMonth() === now.getMonth() && d.getFullYear() === now.getFullYear();
//     });
//     const pending = consults.filter(c => c.status !== 'completed');
//     res.json({
//       total:       completed.length * fee,
//       thisMonth:   thisMonth.length * fee,
//       pending:     pending.length * fee,
//       completed:   completed.length,
//       totalCases:  consults.length,
//       feePerCase:  fee,
//       recentCases: consults.slice(0, 10).map(c => ({
//         id: c._id, crop: c.cropName, status: c.status,
//         date: c.createdAt, amount: c.status === 'completed' ? fee : 0,
//       })),
//     });
//   } catch(e) { res.status(500).json({ error: 'Earnings nahi mile' }); }
// });

// app.get('/api/health', (_, res) =>
//   res.json({ status: 'ok', app: 'BeejHealth API', version: '3.0.0', uptime: process.uptime() }));

// app.get('/api/stats', async (_, res) => {
//   try {
//     const [total, farmers, experts, consultations, pendingConsults] = await Promise.all([
//       dbCount(usersDB, {}),
//       dbCount(usersDB, { type: 'farmer' }),
//       dbCount(usersDB, { type: 'expert' }),
//       dbCount(consultationsDB, {}),
//       dbCount(consultationsDB, { status: 'pending' }),
//     ]);
//     res.json({ total, farmers, experts, consultations, pendingConsults });
//   } catch (e) { res.status(500).json({ error: 'Stats nahi mile' }); }
// });

// /* Dev: all users */
// app.get('/api/admin/users', async (_, res) => {
//   const users = await dbFind(usersDB, {});
//   res.json({
//     count: users.length,
//     users: users.map(u => ({
//       id: u._id, name: u.name, mobile: u.mobile, type: u.type,
//       district: u.district, available: u.available, totalCases: u.totalCases,
//     })),
//   });
// });

// /* Dev: reset all data */
// app.delete('/api/admin/reset', async (_, res) => {
//   await Promise.all([
//     dbRemove(usersDB,         {}, { multi: true }),
//     dbRemove(consultationsDB, {}, { multi: true }),
//     dbRemove(notificationsDB, {}, { multi: true }),
//     dbRemove(chatsDB,         {}, { multi: true }),
//   ]);
//   console.log('🗑️  All data cleared');
//   res.json({ success: true, message: 'All data cleared. Ab seed karo.' });
// });


// /* ═══════════════════════════════════════════════════════════════
//    🤖 ROBOT FLEET APIs
// ═══════════════════════════════════════════════════════════════ */

// /* Helper — seed default robots for a user if none exist */
// async function ensureRobots(userId) {
//   const existing = await dbFind(robotsDB, { ownerId: userId });
//   if (existing.length > 0) return existing;
//   const defaults = [
//     { robotId:'R01', name:'DroneBot Alpha',  type:'Drone',  model:'DJI Agras T40',        status:'online',  battery:87, signal:94, field:'Field 1',    task:'Standby',          emoji:'🚁', sprayArea:0, flights:124, totalArea:'2.1 Acres', ownerId:userId, createdAt:ts() },
//     { robotId:'R02', name:'GroundBot Beta',  type:'Ground', model:'TartanSense TG-1',     status:'online',  battery:62, signal:88, field:'Field 2',    task:'Standby',          emoji:'🤖', sprayArea:0, flights:0,   totalArea:'1.5 Acres', ownerId:userId, createdAt:ts() },
//     { robotId:'R03', name:'DroneBot Gamma',  type:'Drone',  model:'ideaForge RYNO',        status:'offline', battery:12, signal:0,  field:'Charging',   task:'Charging',         emoji:'🚁', sprayArea:0, flights:89,  totalArea:'—',         ownerId:userId, createdAt:ts() },
//     { robotId:'R04', name:'SensorBot Delta', type:'Sensor', model:'Custom IoT v2',         status:'online',  battery:91, signal:99, field:'All Fields', task:'Monitoring',       emoji:'📡', sprayArea:0, flights:0,   totalArea:'4.5 Acres', ownerId:userId, createdAt:ts() },
//   ];
//   return Promise.all(defaults.map(r => dbInsert(robotsDB, r)));
// }

// /* GET /api/robots — list all robots for logged-in user */
// app.get('/api/robots', requireAuth, async (req, res) => {
//   try {
//     const robots = await ensureRobots(req.user.id);
//     // Simulate live battery drain for online robots
//     const live = robots.map(r => ({
//       ...r,
//       battery: r.status === 'online' ? Math.max(10, r.battery - Math.floor(Math.random() * 2)) : r.battery,
//       lastSeen: r.status === 'online' ? 'Just now' : r.lastSeen || '2 hrs ago',
//     }));
//     res.json({ robots: live, total: live.length,
//       online: live.filter(r=>r.status==='online').length,
//       offline: live.filter(r=>r.status==='offline').length,
//     });
//   } catch(e) { console.error(e); res.status(500).json({ error: 'Robots nahi mile' }); }
// });

// /* GET /api/robots/:robotId — single robot detail */
// app.get('/api/robots/:robotId', requireAuth, async (req, res) => {
//   try {
//     let robot = await dbFindOne(robotsDB, { robotId: req.params.robotId, ownerId: req.user.id });
//     if (!robot) {
//       const all = await ensureRobots(req.user.id);
//       robot = all.find(r => r.robotId === req.params.robotId);
//     }
//     if (!robot) return res.status(404).json({ error: 'Robot nahi mila' });
//     res.json({ robot });
//   } catch(e) { res.status(500).json({ error: 'Server error' }); }
// });

// /* PATCH /api/robots/:robotId — update robot status/field/task */
// app.patch('/api/robots/:robotId', requireAuth, async (req, res) => {
//   try {
//     await ensureRobots(req.user.id);
//     const { status, field, task, battery } = req.body;
//     const upd = { updatedAt: ts() };
//     if (status)  upd.status  = status;
//     if (field)   upd.field   = field;
//     if (task)    upd.task    = task;
//     if (battery !== undefined) upd.battery = battery;
//     await dbUpdate(robotsDB, { robotId: req.params.robotId, ownerId: req.user.id }, { $set: upd });
//     const robot = await dbFindOne(robotsDB, { robotId: req.params.robotId, ownerId: req.user.id });
//     // Log the event
//     await dbInsert(robotLogsDB, {
//       robotId: req.params.robotId, ownerId: req.user.id,
//       event: `Status changed to ${status || 'updated'}`, level: 'info',
//       createdAt: ts(),
//     });
//     res.json({ success: true, robot });
//   } catch(e) { res.status(500).json({ error: 'Robot update fail' }); }
// });

// /* POST /api/robots/:robotId/command — send command to robot */
// app.post('/api/robots/:robotId/command', requireAuth, async (req, res) => {
//   try {
//     const { command, params } = req.body;
//     if (!command) return res.status(400).json({ error: 'Command required' });

//     const validCommands = ['start', 'stop', 'pause', 'resume', 'return_home', 'emergency_stop',
//                            'move', 'spray_start', 'spray_stop', 'take_photo', 'wake_up'];
//     if (!validCommands.includes(command))
//       return res.status(400).json({ error: `Unknown command. Valid: ${validCommands.join(', ')}` });

//     // Update robot state based on command
//     const statusMap = {
//       start: 'busy', stop: 'online', pause: 'online',
//       resume: 'busy', return_home: 'busy', emergency_stop: 'online',
//       wake_up: 'online', spray_start: 'busy', spray_stop: 'online',
//     };
//     if (statusMap[command]) {
//       await dbUpdate(robotsDB,
//         { robotId: req.params.robotId, ownerId: req.user.id },
//         { $set: { status: statusMap[command], task: command.replace(/_/g,' '), updatedAt: ts() } }
//       );
//     }

//     // Log command
//     await dbInsert(robotLogsDB, {
//       robotId: req.params.robotId, ownerId: req.user.id,
//       event: `Command: ${command}${params ? ' ' + JSON.stringify(params) : ''}`,
//       level: command === 'emergency_stop' ? 'warning' : 'info',
//       createdAt: ts(),
//     });

//     console.log(`🤖 Robot ${req.params.robotId} ← ${command}`);
//     res.json({ success: true, command, acknowledged: true, timestamp: ts() });
//   } catch(e) { res.status(500).json({ error: 'Command fail' }); }
// });

// /* GET /api/robots/:robotId/logs — activity log */
// app.get('/api/robots/:robotId/logs', requireAuth, async (req, res) => {
//   try {
//     const limit = parseInt(req.query.limit) || 20;
//     let logs = await dbFind(robotLogsDB, { robotId: req.params.robotId, ownerId: req.user.id });
//     logs.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
//     logs = logs.slice(0, limit);

//     // If no logs yet, seed some default activity
//     if (logs.length === 0) {
//       const now = Date.now();
//       logs = [
//         { event: 'System online — all sensors nominal', level: 'info',    createdAt: new Date(now - 1000*60*2).toISOString() },
//         { event: 'GPS lock acquired — 18.59°N 73.74°E', level: 'info',    createdAt: new Date(now - 1000*60*8).toISOString() },
//         { event: 'Battery charge complete — 87%',        level: 'info',    createdAt: new Date(now - 1000*60*18).toISOString() },
//         { event: 'Field 1 spray mission complete',        level: 'info',    createdAt: new Date(now - 1000*60*42).toISOString() },
//         { event: 'Low battery warning — 18%',            level: 'warning', createdAt: new Date(now - 1000*60*70).toISOString() },
//       ];
//     }
//     res.json({ logs, count: logs.length });
//   } catch(e) { res.status(500).json({ error: 'Logs nahi mile' }); }
// });

// /* ═══════════════════════════════════════════════════════════════
//    📡 CAMERA APIs
// ═══════════════════════════════════════════════════════════════ */

// /* GET /api/robots/:robotId/camera — camera stream info */
// app.get('/api/robots/:robotId/camera', requireAuth, async (req, res) => {
//   try {
//     await ensureRobots(req.user.id);
//     const robot = await dbFindOne(robotsDB, { robotId: req.params.robotId, ownerId: req.user.id });
//     if (!robot) return res.status(404).json({ error: 'Robot nahi mila' });

//     const cameras = {
//       Drone:  [{ id:'front',  name:'Front Cam',  type:'RGB',       resolution:'4K/30fps', bitrate:'12 Mbps', latency:'~220ms' },
//                { id:'bottom', name:'Bottom Cam', type:'RGB+IR',    resolution:'4K/30fps', bitrate:'10 Mbps', latency:'~220ms' }],
//       Ground: [{ id:'front',  name:'Front Cam',  type:'RGB',       resolution:'1080p/30fps', bitrate:'8 Mbps', latency:'~180ms' },
//                { id:'soil',   name:'Soil Cam',   type:'Multispect',resolution:'720p/10fps',  bitrate:'4 Mbps', latency:'~250ms' }],
//       Sensor: [{ id:'wide',   name:'Wide Angle', type:'RGB',       resolution:'1080p/15fps', bitrate:'6 Mbps', latency:'~300ms' }],
//     };
//     const cams = cameras[robot.type] || cameras['Drone'];

//     // Latest AI detections (simulated, would come from real ML model)
//     const detections = robot.status !== 'offline' ? [
//       { x:35, y:25, w:15, h:18, label:'Early Blight',  conf:94, color:'#ff4444', action:'Spray needed' },
//       { x:62, y:48, w:12, h:14, label:'Healthy Leaf',  conf:98, color:'#00ff9d', action:'OK'           },
//       { x:18, y:60, w:10, h:12, label:'Pest Damage',   conf:79, color:'#ffd700', action:'Monitor'      },
//     ] : [];

//     res.json({
//       robotId: robot.robotId, robotName: robot.name,
//       isLive: robot.status !== 'offline',
//       cameras: cams,
//       detections,
//       gps: { lat: 18.5913 + Math.random()*0.001, lng: 73.7416 + Math.random()*0.001 },
//       altitude: robot.type === 'Drone' ? `${11 + Math.floor(Math.random()*3)}m` : '0m',
//       speed: robot.type === 'Drone' ? `${3 + (Math.random()*1.5).toFixed(1)}m/s` : `${0.5 + (Math.random()*0.8).toFixed(1)}m/s`,
//     });
//   } catch(e) { res.status(500).json({ error: 'Camera info nahi mili' }); }
// });

// /* POST /api/robots/:robotId/camera/snapshot — save a snapshot */
// app.post('/api/robots/:robotId/camera/snapshot', requireAuth, async (req, res) => {
//   try {
//     const { base64, cameraId } = req.body;
//     const { randomBytes } = await import('crypto');
//     const snapId = randomBytes(6).toString('hex');
//     // Store in uploads map (reuse existing in-memory store)
//     if (base64) uploads.set('snap_' + snapId, { base64, type:'image/jpeg', userId: req.user.id, ts: Date.now() });
//     await dbInsert(robotLogsDB, {
//       robotId: req.params.robotId, ownerId: req.user.id,
//       event: `Snapshot taken — cam:${cameraId||'front'} id:${snapId}`,
//       level: 'info', createdAt: ts(),
//     });
//     res.json({ success: true, snapId, url: base64 ? `/api/upload/photo/snap_${snapId}` : null });
//   } catch(e) { res.status(500).json({ error: 'Snapshot fail' }); }
// });

// /* GET /api/camera/all — all cameras across all robots */
// app.get('/api/camera/all', requireAuth, async (req, res) => {
//   try {
//     const robots = await ensureRobots(req.user.id);
//     const feeds = robots.map(r => ({
//       robotId: r.robotId, robotName: r.name, type: r.type, emoji: r.emoji,
//       field: r.field, status: r.status, isLive: r.status !== 'offline',
//       primaryCam: r.type === 'Drone' ? 'Front 4K' : r.type === 'Ground' ? 'Front 1080p' : 'Wide 1080p',
//     }));
//     res.json({ feeds, total: feeds.length, live: feeds.filter(f=>f.isLive).length });
//   } catch(e) { res.status(500).json({ error: 'Camera feeds nahi mile' }); }
// });

// /* ═══════════════════════════════════════════════════════════════
//    💊 SPRAY SCHEDULER APIs
// ═══════════════════════════════════════════════════════════════ */

// /* GET /api/spray-jobs — all spray jobs */
// app.get('/api/spray-jobs', requireAuth, async (req, res) => {
//   try {
//     let jobs = await dbFind(sprayJobsDB, { ownerId: req.user.id });
//     if (jobs.length === 0) {
//       // Seed default jobs
//       const now = new Date();
//       const seeds = [
//         { jobId:'SJ001', robotId:'R01', field:'Field 1', crop:'Tamatar',   chemical:'Mancozeb 75% WP',      dose:2.5, area:2.0, status:'scheduled', scheduledAt: new Date(now.getTime()+3600000).toISOString(),  priority:'high', disease:'Early Blight',          ownerId:req.user.id, createdAt:ts() },
//         { jobId:'SJ002', robotId:'R01', field:'Field 2', crop:'Gehun',     chemical:'Copper Oxychloride',    dose:3.0, area:1.5, status:'scheduled', scheduledAt: new Date(now.getTime()+86400000).toISOString(), priority:'med',  disease:'Leaf Rust (preventive)', ownerId:req.user.id, createdAt:ts() },
//         { jobId:'SJ003', robotId:'R02', field:'Field 3', crop:'Aalu',      chemical:'Mancozeb 75% WP',      dose:2.5, area:1.0, status:'completed', scheduledAt: new Date(now.getTime()-86400000).toISOString(), priority:'low',  disease:'Late Blight (prev)',     ownerId:req.user.id, createdAt:ts() },
//       ];
//       jobs = await Promise.all(seeds.map(s => dbInsert(sprayJobsDB, s)));
//     }
//     jobs.sort((a,b) => new Date(a.scheduledAt) - new Date(b.scheduledAt));
//     res.json({ jobs, total: jobs.length,
//       pending:   jobs.filter(j=>j.status==='scheduled').length,
//       completed: jobs.filter(j=>j.status==='completed').length,
//     });
//   } catch(e) { res.status(500).json({ error: 'Spray jobs nahi mile' }); }
// });

// /* POST /api/spray-jobs — create new spray job */
// app.post('/api/spray-jobs', requireAuth, async (req, res) => {
//   try {
//     const { robotId, field, crop, chemical, dose, area, scheduledAt, priority, disease } = req.body;
//     if (!field || !chemical || !scheduledAt)
//       return res.status(400).json({ error: 'field, chemical, scheduledAt required' });
//     const { randomBytes } = await import('crypto');
//     const job = await dbInsert(sprayJobsDB, {
//       jobId: 'SJ' + randomBytes(3).toString('hex').toUpperCase(),
//       robotId: robotId || 'R01', field, crop: crop || 'Fasal',
//       chemical, dose: Number(dose) || 2.5, area: Number(area) || 1.0,
//       status: 'scheduled', scheduledAt, priority: priority || 'med',
//       disease: disease || 'Preventive', ownerId: req.user.id, createdAt: ts(),
//     });
//     // Log it
//     await dbInsert(robotLogsDB, {
//       robotId: robotId || 'R01', ownerId: req.user.id,
//       event: `Spray job scheduled — ${field} | ${chemical} | ${new Date(scheduledAt).toLocaleString('en-IN')}`,
//       level: 'info', createdAt: ts(),
//     });
//     console.log(`💊 Spray job created: ${job.jobId} — ${field}`);
//     res.json({ success: true, job });
//   } catch(e) { res.status(500).json({ error: 'Spray job create fail' }); }
// });

// /* PATCH /api/spray-jobs/:jobId — update status */
// app.patch('/api/spray-jobs/:jobId', requireAuth, async (req, res) => {
//   try {
//     const { status } = req.body;
//     await dbUpdate(sprayJobsDB,
//       { jobId: req.params.jobId, ownerId: req.user.id },
//       { $set: { status, updatedAt: ts() } }
//     );
//     const job = await dbFindOne(sprayJobsDB, { jobId: req.params.jobId });
//     res.json({ success: true, job });
//   } catch(e) { res.status(500).json({ error: 'Status update fail' }); }
// });

// /* DELETE /api/spray-jobs/:jobId — cancel spray job */
// app.delete('/api/spray-jobs/:jobId', requireAuth, async (req, res) => {
//   try {
//     await dbRemove(sprayJobsDB, { jobId: req.params.jobId, ownerId: req.user.id });
//     res.json({ success: true, message: 'Spray job cancel ho gaya' });
//   } catch(e) { res.status(500).json({ error: 'Cancel fail' }); }
// });

// /* ═══════════════════════════════════════════════════════════════
//    🗺️ NAVIGATION / MAP APIs  
// ═══════════════════════════════════════════════════════════════ */

// /* GET /api/robots/:robotId/location — live GPS */
// app.get('/api/robots/:robotId/location', requireAuth, async (req, res) => {
//   try {
//     const robot = await dbFindOne(robotsDB, { robotId: req.params.robotId, ownerId: req.user.id });
//     if (!robot) return res.status(404).json({ error: 'Robot nahi mila' });
//     const isLive = robot.status !== 'offline';
//     const baseLat = 18.5913; const baseLng = 73.7416;
//     res.json({
//       robotId: req.params.robotId, isLive,
//       gps: isLive ? {
//         lat:      baseLat + Math.random() * 0.003,
//         lng:      baseLng + Math.random() * 0.003,
//         accuracy: '±2m',
//         altitude: robot.type === 'Drone' ? `${11 + Math.floor(Math.random()*4)}m` : '0m',
//         heading:  Math.floor(Math.random() * 360),
//         speed:    isLive ? `${(Math.random()*4).toFixed(1)}m/s` : '0m/s',
//       } : null,
//       field: robot.field,
//       waypoints: isLive ? [
//         { lat: baseLat + 0.001, lng: baseLng + 0.001, label: 'WP1', done: true },
//         { lat: baseLat + 0.002, lng: baseLng + 0.002, label: 'WP2', done: true },
//         { lat: baseLat + 0.003, lng: baseLng + 0.003, label: 'WP3', done: false },
//         { lat: baseLat + 0.001, lng: baseLng + 0.003, label: 'WP4', done: false },
//       ] : [],
//     });
//   } catch(e) { res.status(500).json({ error: 'Location nahi mili' }); }
// });

// /* POST /api/robots/:robotId/navigate — set mission/waypoints */
// app.post('/api/robots/:robotId/navigate', requireAuth, async (req, res) => {
//   try {
//     const { waypoints, mode, field } = req.body;
//     if (!mode) return res.status(400).json({ error: 'mode required (auto/manual/return)' });
//     await dbUpdate(robotsDB,
//       { robotId: req.params.robotId, ownerId: req.user.id },
//       { $set: { status: 'busy', task: `Navigation: ${mode}`, field: field || 'Field', updatedAt: ts() } }
//     );
//     await dbInsert(robotLogsDB, {
//       robotId: req.params.robotId, ownerId: req.user.id,
//       event: `Mission started — mode:${mode} waypoints:${(waypoints||[]).length}`,
//       level: 'info', createdAt: ts(),
//     });
//     res.json({ success: true, mode, waypointCount: (waypoints||[]).length, eta: '~12 min' });
//   } catch(e) { res.status(500).json({ error: 'Navigation fail' }); }
// });

// /* ═══════════════════════════════════════════════════════════════
//    📊 ROBOT ANALYTICS APIs
// ═══════════════════════════════════════════════════════════════ */

// /* GET /api/robots/analytics/summary — fleet-wide stats */
// app.get('/api/robots/analytics/summary', requireAuth, async (req, res) => {
//   try {
//     const robots  = await ensureRobots(req.user.id);
//     const jobs    = await dbFind(sprayJobsDB, { ownerId: req.user.id });
//     const logs    = await dbFind(robotLogsDB, { ownerId: req.user.id });
//     const completed = jobs.filter(j => j.status === 'completed');
//     const totalArea = completed.reduce((s, j) => s + (j.area || 0), 0);
//     const totalDose = completed.reduce((s, j) => s + (j.dose || 0), 0);
//     res.json({
//       fleetSize:       robots.length,
//       onlineNow:       robots.filter(r => r.status === 'online').length,
//       missionsDone:    completed.length,
//       areaCovered:     `${totalArea.toFixed(1)} Acres`,
//       sprayVolume:     `${(totalDose * totalArea).toFixed(0)}L`,
//       totalFlights:    robots.reduce((s, r) => s + (r.flights || 0), 0),
//       avgBattery:      Math.round(robots.reduce((s, r) => s + r.battery, 0) / (robots.length || 1)),
//       logsToday:       logs.filter(l => new Date(l.createdAt).toDateString() === new Date().toDateString()).length,
//       weeklyData: [
//         { day:'Mon', area: 1.2, spray: 180 },
//         { day:'Tue', area: 0.8, spray: 120 },
//         { day:'Wed', area: 2.1, spray: 315 },
//         { day:'Thu', area: 1.5, spray: 225 },
//         { day:'Fri', area: 1.8, spray: 270 },
//         { day:'Sat', area: 0.6, spray: 90  },
//         { day:'Sun', area: 0.3, spray: 45  },
//       ],
//     });
//   } catch(e) { res.status(500).json({ error: 'Analytics nahi mile' }); }
// });

// /* GET /api/robots/:robotId/maintenance — maintenance status */
// app.get('/api/robots/:robotId/maintenance', requireAuth, async (req, res) => {
//   try {
//     await ensureRobots(req.user.id);
//     const robot = await dbFindOne(robotsDB, { robotId: req.params.robotId, ownerId: req.user.id });
//     if (!robot) return res.status(404).json({ error: 'Robot nahi mila' });
//     const flightHours = Math.round((robot.flights || 0) * 0.45);
//     res.json({
//       robotId: req.params.robotId, name: robot.name,
//       battery: {
//         level:    robot.battery,
//         health:   robot.battery > 50 ? 'Good' : robot.battery > 20 ? 'Fair' : 'Critical',
//         cycles:   robot.flights || 0,
//         nextSwap: robot.battery < 20 ? 'Immediate' : robot.battery < 50 ? 'This week' : 'Next month',
//       },
//       motors: {
//         status: robot.status !== 'offline' ? 'Operational' : 'Unknown',
//         lastCheck: '3 days ago', hoursUsed: flightHours,
//       },
//       sprayer: {
//         tankLevel:  robot.type === 'Drone' ? 75 : 0,
//         nozzleWear: robot.flights > 100 ? 'Replace soon' : 'Good',
//         lastClean:  '2 days ago',
//       },
//       alerts: [
//         ...(robot.battery < 20 ? [{ level:'critical', msg:'Battery critical — charge immediately' }] : []),
//         ...(robot.flights > 120 ? [{ level:'warning',  msg:'Motor inspection due — 124 flights logged' }] : []),
//         ...(robot.signal < 30   ? [{ level:'warning',  msg:'Weak signal — check antenna' }] : []),
//       ],
//       nextService: '7 Nov 2026',
//     });
//   } catch(e) { res.status(500).json({ error: 'Maintenance data nahi mila' }); }
// });


// /* ═══════════════════════════════════════════════════════════════
//    🥥 COCONUT AI ROUTES
// ═══════════════════════════════════════════════════════════════ */

// /* GET /api/coconut/ai-status — AI server online check */
// app.get('/api/coconut/ai-status', async (req, res) => {
//   try {
//     const r = await fetch(`${AI_SERVER}/health`, { signal: AbortSignal.timeout(3000) });
//     const data = await r.json();
//     res.json({ ai_online: true, ...data });
//   } catch {
//     res.json({ ai_online: false, message: 'Python AI server band hai. Port 8000 check karein.' });
//   }
// });

// /* GET /api/coconut/questions — 30 questions info */
// app.get('/api/coconut/questions', (req, res) => {
//   res.json({
//     crop: 'coconut', model: 'EfficientNetV2-S Transfer Learning',
//     total_questions: 30, questions_per_session: 5,
//     classes: [
//       { id: 'Gray Leaf Spot',         hindi: 'Dhusrit Patti Dhabb', severity: 2, urgency: 'medium' },
//       { id: 'Gray Leaf Spot_multiple',hindi: 'Bahut Dhusrit Dhabb', severity: 3, urgency: 'high'   },
//       { id: 'Healthy',                hindi: 'Swasth Ped',          severity: 0, urgency: 'none'   },
//       { id: 'Leaf rot',               hindi: 'Patti Galaav',        severity: 3, urgency: 'high'   },
//       { id: 'Leaf rot_multiple',      hindi: 'Bahut Patti Galaav',  severity: 4, urgency: 'critical'},
//       { id: 'bud rot',                hindi: 'Kali Galaav',         severity: 5, urgency: 'critical'},
//       { id: 'stem bleeding',          hindi: 'Tane se Khoon',       severity: 4, urgency: 'high'   },
//       { id: 'stem bleeding_multiple', hindi: 'Bahut Tane se Khoon', severity: 5, urgency: 'critical'},
//     ],
//     branches: {
//       coconut_spots:  { desc: 'Patti pe daag — Gray Leaf Spot ya Leaf Rot', q_ids: ['cq2','cq3','cq4','cq5'] },
//       coconut_wilt:   { desc: 'Stem Bleeding ya Bud Rot symptoms',          q_ids: ['cq7','cq8','cq9','cq10'] },
//       coconut_yellow: { desc: 'Yellowing — nutrient ya disease',            q_ids: ['cq12','cq13','cq14','cq15'] },
//       coconut_none:   { desc: 'Preventive check — healthy tree',            q_ids: ['cq17','cq18','cq19','cq20'] },
//     },
//   });
// });

// /* POST /api/coconut/questions/session — Q1 answer ke baad 4 questions milte hain */
// app.post('/api/coconut/questions/session', (req, res) => {
//   const { q1_answer } = req.body;
//   const branchMap = { spots:'coconut_spots', wilt:'coconut_wilt', yellow:'coconut_yellow', none:'coconut_none' };
//   const branch = branchMap[q1_answer] || 'coconut_none';

//   const Q1 = {
//     id:'cq1', question:'Aapke naariyal ke ped mein sabse pehle kya symptom dikha?', type:'single',
//     options:[
//       {id:'spots',  label:'🟤 Patti pe daag ya dhabba',          desc:'Brown, gray ya kala daag'},
//       {id:'wilt',   label:'🔴 Tana ya kali mein problem',         desc:'Trunk se liquid ya kali gal rahi'},
//       {id:'yellow', label:'🟡 Patti peeli ho rahi hai',           desc:'Leaves yellow ya dry'},
//       {id:'none',   label:'✅ Koi symptom nahi, preventive check',desc:'Tree healthy lagta hai'},
//     ],
//   };

//   const ALL_BRANCH_QS = {
//     coconut_spots: [
//       {id:'cq2', question:'Daag ka rang kaisa hai?',
//        options:[{id:'gray',label:'⬜ Gray/Ash rang'},{id:'brown',label:'🟫 Brown/Dark Brown'},{id:'yellow_edge',label:'🟡 Brown with yellow border'},{id:'black',label:'⬛ Kala daag'}]},
//       {id:'cq3', question:'Daag patti ke kahan par hai?',
//        options:[{id:'tip',label:'🔝 Patti ki nauk pe'},{id:'middle',label:'↔️ Beech mein'},{id:'base',label:'⬇️ Neeche jaad ke paas'},{id:'all',label:'📍 Poori patti pe'}]},
//       {id:'cq4', question:'Kitni pattiyon pe daag hai?',
//        options:[{id:'one',label:'1️⃣ Ek ya do patti'},{id:'few',label:'🔢 3-5 pattiyan'},{id:'many',label:'📊 Adhi se zyada'},{id:'all',label:'⚠️ Saari pattiyan'}]},
//       {id:'cq5', question:'Daag ka size kaisa hai?',
//        options:[{id:'small',label:'🔵 Chota (1-2mm)'},{id:'medium',label:'🟡 Medium (5-10mm)'},{id:'large',label:'🔴 Bada (>1cm)'},{id:'merge',label:'💀 Daag aapas mein jud gaye'}]},
//     ],
//     coconut_wilt: [
//       {id:'cq7', question:'Problem mainly kahan hai?',
//        options:[{id:'trunk',label:'🌴 Tane (trunk) pe'},{id:'bud',label:'🌱 Nai kali/growth pe'},{id:'roots',label:'🌿 Neeche jaad ke paas'},{id:'both',label:'⚠️ Trunk aur kali dono'}]},
//       {id:'cq8', question:'Tane se koi liquid nikal raha hai?',
//        options:[{id:'dark_liquid',label:'🩸 Haan, dark brown/red liquid'},{id:'clear',label:'💧 Haan, transparent liquid'},{id:'no',label:'❌ Nahi, koi liquid nahi'},{id:'smell',label:'👃 Liquid hai + buri smell'}]},
//       {id:'cq9', question:'Nai patti ya kali ka kya haal hai?',
//        options:[{id:'normal',label:'✅ Normal badi ho rahi hai'},{id:'brown',label:'🟫 Kali brown/black ho gayi'},{id:'small',label:'📉 Patti choti reh gayi'},{id:'dead',label:'💀 Kali/patti mar gayi'}]},
//       {id:'cq10', question:'Yeh problem kitne samay se hai?',
//        options:[{id:'week',label:'📅 1 hafte se kam'},{id:'month',label:'🗓️ 1-2 mahine'},{id:'old',label:'📆 3-6 mahine'},{id:'very_old',label:'⏳ 6 mahine se zyada'}]},
//     ],
//     coconut_yellow: [
//       {id:'cq12', question:'Yellowing kahan se shuru hua?',
//        options:[{id:'old_leaves',label:'⬇️ Neeche ki purani pattiyon se'},{id:'new_leaves',label:'⬆️ Upar ki nai pattiyon se'},{id:'all_at_once',label:'📊 Saari pattiyan ek saath'},{id:'patches',label:'🗺️ Kuch jagah kuch jagah'}]},
//       {id:'cq13', question:'Patti ki nasubon (veins) ka rang kaisa hai?',
//        options:[{id:'green_vein',label:'🟢 Nasuben hari, baaki peeli'},{id:'all_yellow',label:'🟡 Sab kuch peela'},{id:'brown_vein',label:'🟫 Nasuben bhi brown'},{id:'normal',label:'✅ Nasuben theek hain'}]},
//       {id:'cq14', question:'Paani dene ka schedule kya hai?',
//        options:[{id:'too_much',label:'💧 Bahut zyada paani'},{id:'less',label:'🏜️ Kam paani milta hai'},{id:'regular',label:'✅ Regular schedule pe'},{id:'rain',label:'🌧️ Sirf baarish pe depend'}]},
//       {id:'cq15', question:'Khaad (fertilizer) kab diya tha?',
//        options:[{id:'recent',label:'📅 1 mahine ke andar'},{id:'months',label:'🗓️ 2-6 mahine pehle'},{id:'long',label:'📆 6+ mahine nahi diya'},{id:'never',label:'❌ Kabhi nahi diya'}]},
//     ],
//     coconut_none: [
//       {id:'cq17', question:'Aap kaun sa coconut variety ughate hain?',
//        options:[{id:'tall',label:'🌴 Tall variety (Tiptur, West Coast)'},{id:'dwarf',label:'🌿 Dwarf variety'},{id:'hybrid',label:'🔬 Hybrid variety'},{id:'unknown',label:'❓ Pata nahi'}]},
//       {id:'cq18', question:'Aapka farm kaisi jagah pe hai?',
//        options:[{id:'coastal',label:'🌊 Coastal (samundar ke paas)'},{id:'inland',label:'🏔️ Inland (andar ki taraf)'},{id:'humid',label:'💧 Bahut nami wali jagah'},{id:'dry',label:'☀️ Sukhha ilaqa'}]},
//       {id:'cq19', question:'Last spray kab kiya tha?',
//        options:[{id:'recent',label:'📅 1 mahine ke andar'},{id:'months',label:'🗓️ 2-4 mahine pehle'},{id:'long',label:'📆 6+ mahine nahi kiya'},{id:'never',label:'❌ Kabhi nahi'}]},
//       {id:'cq20', question:'Aas paas ke pedo mein koi bimari dikhi?',
//        options:[{id:'yes',label:'⚠️ Haan, bahut pedo mein disease hai'},{id:'no',label:'✅ Nahi, sab theek hain'},{id:'some',label:'🔢 Kuch pedo mein thodi problem'},{id:'unknown',label:'❓ Maine check nahi kiya'}]},
//     ],
//   };

//   const branchQs = ALL_BRANCH_QS[branch] || ALL_BRANCH_QS.coconut_none;
//   res.json({ success:true, branch, q1_answer, total_questions:5, questions:[Q1, ...branchQs] });
// });


// /* POST /api/coconut/quick-scan — Fast pre-analysis for question selection */
// app.post('/api/coconut/quick-scan', async (req, res) => {
//   try {
//     const { photoBase64 } = req.body;
//     if (!photoBase64) return res.status(400).json({ error: 'photoBase64 required' });
//     const aiResult = await callAI('/predict', {
//       image_base64: photoBase64,
//       consultation_id: null,
//       question_answers: {},
//     });
//     // Return minimal result for question selection
//     res.json({
//       disease:    aiResult.disease,
//       confidence: aiResult.confidence,
//       severity:   aiResult.severity,
//       is_healthy: aiResult.is_healthy,
//       urgency:    aiResult.urgency,
//       top3:       aiResult.top3,
//     });
//   } catch(e) {
//     if (e.message === 'AI_SERVER_DOWN') return res.status(503).json({ error: 'AI server band hai' });
//     res.status(500).json({ error: e.message });
//   }
// });

// /* POST /api/coconut/analyze — Photo → AI detection → save to DB */
// app.post('/api/coconut/analyze', requireAuth, async (req, res) => {
//   try {
//     const { photoBase64, photoId, consultationId, questionAnswers } = req.body;
//     const photoIds = Array.isArray(req.body.photoIds) ? req.body.photoIds.filter(Boolean) : (photoId ? [photoId] : []);
//     const photoUrls = Array.isArray(req.body.photoUrls) ? req.body.photoUrls.filter(Boolean) : [];
//     const resolvedPhotoCount = Math.max(
//       Number(req.body.photoCount) || 0,
//       photoIds.length,
//       photoUrls.length,
//       photoId ? 1 : 0,
//       photoBase64 ? 1 : 0,
//     );
//     if (!photoBase64 && !photoId)
//       return res.status(400).json({ error: 'photoBase64 ya photoId dena hoga' });
//     if (resolvedPhotoCount > MAX_REPORT_PHOTOS)
//       return res.status(400).json({ error: `Maximum ${MAX_REPORT_PHOTOS} photos allowed per consultation` });
//     if (resolvedPhotoCount < MIN_REPORT_PHOTOS)
//       return res.status(400).json({ error: `Minimum ${MIN_REPORT_PHOTOS} photos required for report generation` });

//     let base64ToSend = photoBase64;
//     if (photoId && !photoBase64) {
//       const photoDoc = await dbFindOne(photosDB, { photoId, userId: req.user.id });
//       if (!photoDoc) return res.status(404).json({ error: 'Photo nahi mili' });
//       base64ToSend = photoDoc.base64;
//     }

//     // Call Python AI Server
//     console.log(`🥥 Coconut AI analyze — user: ${req.user.id}`);
//     let aiResult;
//     try {
//       aiResult = await callAI('/predict', {
//         image_base64: base64ToSend,
//         consultation_id: consultationId || null,
//         question_answers: questionAnswers || {},
//       });
//     } catch(aiErr) {
//       if (aiErr.message === 'AI_SERVER_DOWN') {
//         return res.status(503).json({
//           error: 'AI server band hai',
//           message: 'Python AI server start karein: cd ai_service && python ai_server.py',
//           ai_down: true,
//         });
//       }
//       throw aiErr;
//     }

//     // Save to DB
//     let consultation;
//     if (consultationId) {
//       await dbUpdate(consultationsDB,
//         { _id: consultationId, farmerId: req.user.id },
//         { $set: {
//           disease: aiResult.disease, disease_hindi: aiResult.disease_hindi,
//           confidence: aiResult.confidence, severity: aiResult.severity,
//           is_healthy: aiResult.is_healthy, urgency: aiResult.urgency,
//           ai_description: aiResult.description, ai_treatments: aiResult.treatments,
//           ai_top3: aiResult.top3, ai_model: aiResult.model_version,
//           coconut_only: true,
//           method: 'photo',
//           photoUploaded: true,
//           photoId: photoIds[0] || photoId || null,
//           photoIds,
//           photoUrls,
//           photoCount: resolvedPhotoCount,
//           updatedAt: ts(),
//         }}
//       );
//       await linkPhotosToConsultation(photoIds, consultationId, req.user.id);
//       consultation = await dbFindOne(consultationsDB, { _id: consultationId });
//     } else {
//       const experts = await dbFind(usersDB, { type: 'expert', available: true });
//       const expert  = experts.length > 0 ? experts[Math.floor(Math.random() * experts.length)] : null;
//       consultation  = await dbInsert(consultationsDB, {
//         farmerId: req.user.id, expertId: expert?._id || null,
//         expertName: expert?.name || 'Pending Assignment',
//         cropId: 'coconut', cropName: 'Naariyal (Coconut)', cropEmoji: '🥥',
//         method: 'photo', photoUploaded: true, photoId: photoIds[0] || photoId || null,
//         photoIds,
//         photoUrls,
//         photoCount: resolvedPhotoCount,
//         disease: aiResult.disease, disease_hindi: aiResult.disease_hindi,
//         confidence: aiResult.confidence, severity: aiResult.severity,
//         is_healthy: aiResult.is_healthy, urgency: aiResult.urgency,
//         ai_description: aiResult.description, ai_treatments: aiResult.treatments,
//         ai_top3: aiResult.top3, ai_model: aiResult.model_version,
//         coconut_only: true, answers: questionAnswers || {},
//         status: expert ? 'expert_assigned' : 'pending', report: null,
//         createdAt: ts(), updatedAt: ts(),
//       });
//       // Notify farmer
//       await dbInsert(notificationsDB, {
//         userId: req.user.id, type: 'consultation', icon: '🥥',
//         title: `AI Report Ready — ${aiResult.disease}`,
//         body: `${aiResult.disease} detected (${aiResult.confidence}% confidence). ${aiResult.is_healthy ? '🌴 Tree healthy hai!' : aiResult.urgency === 'critical' ? '⚠️ Turant action zaroor!' : '📋 Treatment plan ready.'}`,
//         read: false, consultationId: consultation._id, createdAt: ts(),
//       });
//       if (expert) {
//         await dbInsert(notificationsDB, {
//           userId: expert._id, type: 'new_case', icon: '🥥',
//           title: 'Naya Coconut Case',
//           body: `${aiResult.disease} detected — farmer ka case review karein.`,
//           read: false, consultationId: consultation._id, createdAt: ts(),
//         });
//         await dbUpdate(usersDB, { _id: expert._id }, { $inc: { totalCases: 1 } });
//       }
//       await linkPhotosToConsultation(photoIds, consultation._id, req.user.id);
//     }

//     console.log(`✅ Coconut AI: ${aiResult.disease} (${aiResult.confidence}%) → ${consultation._id}`);
//     res.json({ success: true, consultation_id: consultation._id, ai_result: aiResult, consultation });

//   } catch(e) {
//     console.error('Coconut analyze error:', e);
//     res.status(500).json({ error: e.message });
//   }
// });

// /* GET /api/coconut/consultations — Sirf coconut consultations */
// app.get('/api/coconut/consultations', requireAuth, async (req, res) => {
//   try {
//     const q = { cropId: 'coconut',
//       ...(req.user.type === 'expert' ? { expertId: req.user.id } : { farmerId: req.user.id }) };
//     const list = await dbFind(consultationsDB, q);
//     list.sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt));
//     res.json({ consultations: list, count: list.length });
//   } catch(e) { res.status(500).json({ error: 'Coconut consultations nahi mile' }); }
// });

// /* POST /api/support — dedicated support ticket endpoint */
// app.post('/api/support', async (req, res) => {
//   try {
//     const { name, mobile, issue, desc } = req.body;
//     if (!issue?.trim() || !desc?.trim())
//       return res.status(400).json({ error: 'Issue aur description required hai' });
//     const ticket = await dbInsert(consultationsDB, {
//       farmerId: req.user?.id || 'guest',
//       expertId: null, expertName: 'Support Team',
//       cropId: 'support', cropName: `Support: ${issue}`, cropEmoji: '🆘',
//       method: 'support', disease: issue, confidence: 100, severity: 1,
//       status: 'pending', report: null,
//       answers: { name: name || '', mobile: mobile || '', desc },
//       createdAt: ts(), updatedAt: ts(),
//     });
//     console.log(`🆘 Support ticket: ${issue} from ${name || 'guest'}`);
//     res.json({ success: true, ticketId: ticket._id, message: '24 ghante mein reply milegi' });
//   } catch (e) { res.status(500).json({ error: 'Ticket save nahi hua' }); }
// });

// /* 404 */
// app.use('/api/*', (req, res) => res.status(404).json({ error: `Route nahi mila: ${req.path}` }));

// /* Error handler */
// app.use((err, _req, res, _next) => {
//   console.error('Unhandled:', err);
//   res.status(500).json({ error: 'Internal server error' });
// });

// /* ── Start ── */
// app.listen(PORT, () => {
//   console.log('\n🌱 ════════════════════════════════════════');
//   console.log('   BeejHealth API v3.0 chal raha hai!');
//   console.log(`   🚀  http://localhost:${PORT}/api/health`);
//   console.log(`   📊  http://localhost:${PORT}/api/stats`);
//   console.log(`   👥  http://localhost:${PORT}/api/admin/users`);
//   console.log('🌱 ════════════════════════════════════════\n');
// });
