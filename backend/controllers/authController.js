import bcrypt from 'bcryptjs';
import { prisma } from '../db.js';
import { normalizeMobile, findUsersByMobile, findUserByMobile, ts, safe, sign } from '../utils/helpers.js';

export const sendOtp = async (req, res) => {
  try {
    const { mobile } = req.body;
    const cleanMobile = normalizeMobile(mobile);
    if (cleanMobile.length < 10)
      return res.status(400).json({ error: 'Valid 10-digit mobile number daalein' });
    console.log(`📱 OTP [demo] → ${cleanMobile} : 123456`);
    res.json({ success: true, message: 'OTP bheja gaya! (Demo mode — OTP hai: 123456)' });
  } catch (e) { res.status(500).json({ error: 'Server error' }); }
};

export const login = async (req, res) => {
  try {
    const { mobile, password, otp, method, type } = req.body;
    const cleanMobile = normalizeMobile(mobile);
    if (cleanMobile.length < 10) return res.status(400).json({ error: 'Mobile number daalein' });

    const candidates = await findUsersByMobile(cleanMobile, type);
    if (!candidates.length) return res.status(404).json({ error: 'Yeh mobile registered nahi. Pehle register karein.' });

    let user = null;
    if (method === 'otp') {
      if (!otp || String(otp).length < 6)
        return res.status(400).json({ error: '6-digit OTP enter karein' });
      user = candidates[0];
    } else {
      if (!password) return res.status(400).json({ error: 'Password enter karein' });
      for (const candidate of candidates) {
        if (!candidate?.password) continue;
        const ok = await bcrypt.compare(String(password), candidate.password);
        if (ok) {
          user = candidate;
          break;
        }
      }
      if (!user) return res.status(401).json({ error: 'Password galat hai. Dobara try karein.' });
    }

    await prisma.user.update({
      where: { id: user.id || user._id },
      data: { lastLogin: new Date() }
    });
    console.log(`✅ Login: ${user.name} (${user.type})`);
    res.json({ success: true, token: sign(user), user: safe(user) });
  } catch (e) { console.error('login:', e); res.status(500).json({ error: 'Server error. Dobara try karein.' }); }
};

export const register = async (req, res) => {
  try {
    const { name, mobile, email, password, type,
            district, taluka, village, soil,
            spec, fee, university, langs, crops } = req.body;

    if (!name || name.trim().length < 2) return res.status(400).json({ error: 'Naam 2+ characters hona chahiye' });
    const cleanMobile = normalizeMobile(mobile);
    if (cleanMobile.length < 10) return res.status(400).json({ error: 'Valid 10-digit mobile daalein' });
    if (!password || password.length < 8) return res.status(400).json({ error: 'Password 8+ characters hona chahiye' });
    if (type === 'expert' && !spec) return res.status(400).json({ error: 'Specialization daalein' });

    const exists = await findUserByMobile(cleanMobile);
    if (exists) return res.status(409).json({ error: 'Yeh mobile pehle se registered hai. Login karein.' });

    const trimName = name.trim();
    const initials = trimName.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
    const hashed = await bcrypt.hash(password, 12);

    const user = await prisma.user.create({
      data: {
        name: trimName, mobile: cleanMobile, email: email?.trim() || '', password: hashed, type: type || 'farmer',
        initials, state: req.body.state || 'Maharashtra', district: district || '', taluka: req.body.taluka || '',
        village: village || '', soil: soil || '', farmSize: Number(req.body.farmSize) || 0,
        irrigation: req.body.irrigation || '', spec: spec || '', fee: Number(fee) || 0,
        university: university || '', langs: langs || 'Hindi', expYrs: Number(req.body.expYrs) || 0,
        crops: Array.isArray(crops) ? crops : [], verified: false, available: type === 'expert', rating: 0,
        totalCases: 0, bio: ''
      }
    });

    await prisma.notification.create({
      data: {
        userId: user.id, type: 'welcome', icon: '🌱', title: 'BeejHealth mein Swagat! 🎉',
        body: `Namaste ${trimName}! Aapka account ban gaya hai. ${type === 'expert' ? 'Ab aap cases receive kar sakte hain.' : 'Ab aap crop consultation le sakte hain.'}`,
        read: false
      }
    });

    console.log(`🆕 Register: ${trimName} (${user.type})`);
    res.json({ success: true, token: sign(user), user: { ...safe(user), fresh: true } });
  } catch (e) {
    if (e.code === 'P2002') return res.status(409).json({ error: 'Mobile already registered' });
    console.error('register:', e);
    res.status(500).json({ error: 'Registration fail hua. Console check karein.' });
  }
};

export const getMe = async (req, res) => {
  try {
    const user = await prisma.user.findUnique({ where: { id: req.user.id } });
    if (!user) return res.status(404).json({ error: 'User nahi mila' });
    res.json({ user: safe(user) });
  } catch (e) { res.status(500).json({ error: 'Server error' }); }
};

export const updateProfile = async (req, res) => {
  try {
    const allowed = ['name','email','state','district','taluka','village','soil',
                     'farmSize','irrigation','spec','fee','langs','crops','available','university','bio','expYrs'];
    const upd = {};
    allowed.forEach(k => { if (req.body[k] !== undefined) upd[k] = req.body[k]; });
    if (upd.name) upd.initials = upd.name.split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
    if (upd.fee) upd.fee = Number(upd.fee);
    if (upd.farmSize) upd.farmSize = Number(upd.farmSize);
    if (upd.expYrs) upd.expYrs = Number(upd.expYrs);
    
    const user = await prisma.user.update({
      where: { id: req.user.id },
      data: upd
    });
    res.json({ success: true, user: safe(user) });
  } catch (e) { res.status(500).json({ error: 'Profile update fail' }); }
};

export const updatePassword = async (req, res) => {
  try {
    const { oldPassword, newPassword } = req.body;
    if (!newPassword || newPassword.length < 8)
      return res.status(400).json({ error: 'Naya password 8+ characters' });
    const user = await prisma.user.findUnique({ where: { id: req.user.id } });
    const ok = await bcrypt.compare(oldPassword, user.password);
    if (!ok) return res.status(401).json({ error: 'Purana password galat hai' });
    const hashed = await bcrypt.hash(newPassword, 12);
    await prisma.user.update({
      where: { id: req.user.id },
      data: { password: hashed }
    });
    res.json({ success: true, message: 'Password successfully change ho gaya!' });
  } catch (e) { res.status(500).json({ error: 'Password update fail' }); }
};
