import jwt from 'jsonwebtoken';
import { readFileSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { JWT_SECRET, AI_SERVER } from '../config/env.js';
import { prisma } from '../db.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const ts = () => new Date().toISOString();
export const safe = (u) => { if (!u) return null; const { password: _, ...r } = u; return r; };
export const sign = (u) => jwt.sign({ id: u.id || u._id, mobile: u.mobile, type: u.type }, JWT_SECRET, { expiresIn: '30d' });
export const normalizeMobile = (mobile) => String(mobile ?? '').replace(/\D/g, '').slice(-10);
export const toMillis = (value) => {
  const ms = new Date(value || 0).getTime();
  return Number.isFinite(ms) ? ms : 0;
};

export async function findUsersByMobile(rawMobile, preferredType = null) {
  const mobile = normalizeMobile(rawMobile);
  if (mobile.length < 10) return [];

  let users = [];
  try {
    users = await prisma.user.findMany({ where: { mobile } });
  } catch (e) {
    console.warn('users lookup failed:', e.message);
  }

  return users.sort((a, b) => {
    if (preferredType) {
      const aPref = a.type === preferredType;
      const bPref = b.type === preferredType;
      if (aPref !== bPref) return aPref ? -1 : 1;
    }
    const aTime = new Date(a.lastLogin || a.createdAt || 0).getTime();
    const bTime = new Date(b.lastLogin || b.createdAt || 0).getTime();
    if (bTime !== aTime) return bTime - aTime;
    return String(b.id || '').localeCompare(String(a.id || ''));
  });
}

export async function findUserByMobile(rawMobile) {
  const users = await findUsersByMobile(rawMobile);
  return users[0] || null;
}

export async function enrichConsultationWithPhotos(consultation) {
  if (!consultation) return consultation;

  const existingPhotoIds = Array.isArray(consultation.photoIds)
    ? consultation.photoIds.filter(Boolean)
    : consultation.photoId ? [consultation.photoId] : [];
  const existingPhotoUrls = Array.isArray(consultation.photoUrls)
    ? consultation.photoUrls.filter(Boolean)
    : consultation.photoUrl ? [consultation.photoUrl] : [];
  
  let reportSnapshot = consultation.reportSnapshot;
  if (typeof reportSnapshot === 'string') {
    try { reportSnapshot = JSON.parse(reportSnapshot); } catch (e) {}
  }
  reportSnapshot = reportSnapshot || {};
  
  const snapshotPhotos = Array.isArray(reportSnapshot?.photos)
    ? reportSnapshot.photos.filter((photo) => photo?.url || photo?.photoId)
    : [];

  if (existingPhotoIds.length || existingPhotoUrls.length || snapshotPhotos.length) {
    return consultation;
  }

  const consultationTime = toMillis(consultation.createdAt);
  const candidatePhotos = await prisma.photo.findMany({ where: { userId: consultation.farmerId } }).catch(() => []);
  if (!candidatePhotos.length) return consultation;

  const matched = candidatePhotos
    .filter((photo) => {
      if (photo.consultationId === consultation.id || photo.consultationId === consultation._id) return true;
      if (photo.consultationId && photo.consultationId !== consultation.id && photo.consultationId !== consultation._id) return false;
      const photoTime = toMillis(photo.createdAt);
      const diff = Math.abs(photoTime - consultationTime);
      return diff <= 6 * 60 * 60 * 1000;
    })
    .sort((a, b) => {
      const aExact = (a.consultationId === consultation.id || a.consultationId === consultation._id) ? 0 : 1;
      const bExact = (b.consultationId === consultation.id || b.consultationId === consultation._id) ? 0 : 1;
      if (aExact !== bExact) return aExact - bExact;
      return toMillis(a.createdAt) - toMillis(b.createdAt);
    })
    .slice(0, 5);

  if (!matched.length) return consultation;

  const photoIds = matched.map((photo) => photo.photoId).filter(Boolean);
  const photoUrls = matched.map((photo) => photo.url || `/api/photos/${photo.photoId}`).filter(Boolean);
  const photoCount = matched.length;
  
  reportSnapshot.photos = matched.map((photo, index) => ({
    photoId: photo.photoId || null,
    url: photo.url || `/api/photos/${photo.photoId}`,
    index: index + 1,
  }));

  const updateData = {
      photoUploaded: true,
      photoId: photoIds[0] || null,
      photoUrl: photoUrls[0] || null,
      photoIds,
      photoUrls,
      photoCount,
      reportSnapshot,
      updatedAt: new Date(),
  };

  await prisma.consultation.update({
    where: { id: consultation.id || consultation._id },
    data: updateData
  }).catch(() => {});

  if (photoIds.length) {
    await prisma.photo.updateMany({
      where: { photoId: { in: photoIds }, userId: consultation.farmerId },
      data: { consultationId: consultation.id || consultation._id }
    }).catch(() => {});
  }

  return {
    ...consultation,
    ...updateData,
    updatedAt: new Date().toISOString()
  };
}

export async function callAI(endpoint, body) {
  try {
    const res = await fetch(`${AI_SERVER}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30000),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'AI server error' }));
      throw new Error(err.detail || err.error || `AI ${res.status}`);
    }
    return res.json();
  } catch(e) {
    if (e.code === 'ECONNREFUSED' || e.name === 'TimeoutError') {
      throw new Error('AI_SERVER_DOWN');
    }
    throw e;
  }
}
