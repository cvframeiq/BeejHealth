# 🌱 BeejHealth v3.0 — Production Ready

## Quick Start

```bash
# 1. Unzip
cd bh_prod

# 2. All dependencies
npm install && npm run setup

# 3. Seed dummy data (pehli baar)
npm run seed

# 4. Start both servers
npm run dev
```

**Frontend:** http://localhost:5173  
**Backend API:** http://localhost:3000/api/health

---

## 🔑 Login Credentials

### Farmers
| Mobile | Password | District |
|--------|----------|---------|
| 9876543210 | farmer123 | Pune |
| 9876543211 | farmer123 | Nashik |
| 9876543212 | farmer123 | Solapur |
| 9876543213 | farmer123 | Nagpur |
| 9876543214 | farmer123 | Aurangabad |

### Experts  
| Mobile | Password | Specialization | Status |
|--------|----------|----------------|--------|
| 9000000001 | expert123 | Plant Pathologist | 🟢 Online |
| 9000000002 | expert123 | Horticulture Expert | 🟢 Online |
| 9000000003 | expert123 | Soil Scientist | 🔴 Offline |
| 9000000004 | expert123 | Crop Scientist | 🟢 Online |
| 9000000005 | expert123 | Horticulture Expert | 🔴 Offline |
| 9000000006 | expert123 | Crop Scientist | 🟢 Online |

> **OTP Login:** Any 6-digit number works (demo mode: 123456)

---

## ✅ All Flows Working

### Farmer Flow
1. **Register/Login** → Data DB mein store, JWT token localStorage mein
2. **Crop Consultation** → 4 steps: Crop → Photo → 5 Adaptive Questions → AI Report
3. **AI Report** → FrameIQ branded report, disease + treatment plan
4. **Expert Booking** → Expert select → Payment flow → Booking confirm
5. **Chat** → Real-time messages with expert (stored in DB)
6. **My Consultations** → All past cases, status, report view
7. **Notifications** → Disease alerts, expert replies, reminders
8. **Profile/Settings** → Edit name, email, village — saves to DB
9. **Logout** → Session clear, redirect to home

### Expert Flow
1. **Login** → Expert dashboard with assigned cases
2. **Cases** → List all assigned consultations
3. **Case Detail** → View farmer's crop, AI analysis, answers
4. **Chat** → Message farmer, stored in DB
5. **Submit Report** → Write report → Farmer ko notify
6. **Availability** → Toggle online/offline (API se update)
7. **Earnings** → Dashboard

---

## 📁 Structure

```
bh_prod/
├── package.json           ← Root (concurrently)
├── backend/
│   ├── server.js          ← Express API (port 3000)
│   ├── db.js              ← NeDB (4 collections)
│   ├── seed.js            ← Dummy data
│   ├── .env.example       ← Environment variables
│   └── data/              ← Auto-created DB files
└── frontend/
    ├── src/BeejHealth.jsx ← Complete React app
    ├── src/main.jsx
    ├── vite.config.js     ← Proxy /api → :3000
    └── index.html
```

---

## 🔌 API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | /api/auth/send-otp | ❌ | OTP send |
| POST | /api/auth/login | ❌ | Login |
| POST | /api/auth/register | ❌ | Register |
| GET | /api/auth/me | ✅ | Current user |
| PATCH | /api/auth/profile | ✅ | Update profile |
| PATCH | /api/auth/password | ✅ | Change password |
| GET | /api/experts | ❌ | List experts |
| PATCH | /api/experts/availability | ✅ Expert | Toggle online |
| POST | /api/consultations | ✅ | New consultation |
| GET | /api/consultations | ✅ | My consultations |
| GET | /api/consultations/:id | ✅ | Detail |
| PATCH | /api/consultations/:id/status | ✅ | Update status/report |
| GET | /api/consultations/:id/messages | ✅ | Chat messages |
| POST | /api/consultations/:id/messages | ✅ | Send message |
| GET | /api/notifications | ✅ | My notifications |
| PATCH | /api/notifications/read-all | ✅ | Mark all read |
| GET | /api/health | ❌ | Health check |
| GET | /api/stats | ❌ | Platform stats |
| GET | /api/admin/users | ❌ | Dev: all users |
| DELETE | /api/admin/reset | ❌ | Dev: clear data |

---

## 🗄️ Database (NeDB)

4 collections stored in `backend/data/`:
- **users.db** — Farmers + Experts (bcrypt passwords, JWT tokens)
- **consultations.db** — Crop diagnosis sessions
- **notifications.db** — All notifications
- **chats.db** — Chat messages

Data persists between restarts automatically.

---

## 🔧 Production Deploy

```bash
# Build frontend
npm run build

# Start backend (serves API only)
npm start
```

Set in `backend/.env`:
```
PORT=3000
JWT_SECRET=your-secure-secret-here
FRONTEND_URL=https://your-domain.com
NODE_ENV=production
```
