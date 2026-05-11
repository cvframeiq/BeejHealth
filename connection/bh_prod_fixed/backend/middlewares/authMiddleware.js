import jwt from 'jsonwebtoken';
import { JWT_SECRET } from '../config/env.js';
import { prisma } from '../db.js';

export async function requireAuth(req, res, next) {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) return res.status(401).json({ error: 'Token nahi mila. Login karein.' });
  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    let user = decoded?.id ? await prisma.user.findUnique({ where: { id: decoded.id } }).catch(() => null) : null;
    if (!user && decoded?.mobile) {
      user = await prisma.user.findFirst({ where: { mobile: String(decoded.mobile) } }).catch(() => null);
    }
    req.user = user
      ? { ...decoded, id: user.id, mobile: user.mobile, type: user.type, name: user.name }
      : decoded;
    next();
  } catch (e) {
    res.status(401).json({ error: 'Session expire ho gaya. Dobara login karein.' });
  }
}

export function requireExpert(req, res, next) {
  if (req.user?.type !== 'expert') return res.status(403).json({ error: 'Sirf experts ke liye.' });
  next();
}
