import { randomBytes } from 'crypto';
import { prisma } from '../db.js';
import { uploads, MAX_PHOTO_UPLOAD_BYTES } from '../utils/memoryStore.js';

export const uploadPhoto = async (req, res) => {
  try {
    const { base64, type = 'image/jpeg', consultationId, index = 0, totalExpected = 1 } = req.body;
    if (!base64) return res.status(400).json({ error: 'Image data missing' });
    if (!base64.startsWith('data:image'))
      return res.status(400).json({ error: 'Valid image chahiye (JPG/PNG/HEIC)' });

    const sizeBytes = Math.round((base64.length * 3) / 4);
    if (sizeBytes > MAX_PHOTO_UPLOAD_BYTES)
      return res.status(400).json({ error: 'Ek photo max 25MB honi chahiye' });

    if (consultationId) {
      const existing = await prisma.photo.findMany({ where: { consultationId, userId: req.user.id } });
      if (existing.length >= 5)
        return res.status(400).json({ error: 'Maximum 5 photos allowed per consultation' });
    }

    const photoId = randomBytes(8).toString('hex');

    uploads.set(photoId, { base64, type, userId: req.user.id });
    try {
      await prisma.photo.create({
        data: {
          photoId, userId: req.user.id, consultationId: consultationId || null,
          type, base64, index: Number(index), totalExpected: Number(totalExpected),
          sizeKB: Math.round(sizeBytes / 1024), url: `/api/photos/${photoId}`
        }
      });
    } catch (dbErr) {
      if (dbErr?.code !== 'P2003') throw dbErr;
      console.warn(`Photo DB save skipped for stale user ${req.user.id}; keeping upload in memory: ${photoId}`);
      return res.json({ success: true, photoId, url: `/api/photos/${photoId}`, sizeKB: Math.round(sizeBytes / 1024), index });
    }

    if (consultationId) {
      const existing = await prisma.photo.findMany({ where: { consultationId, userId: req.user.id }, orderBy: { index: 'asc' } });
      const photoIds = existing.map(p => p.photoId);
      const photoUrls = existing.map(p => p.url);
      await prisma.consultation.update({
        where: { id: consultationId },
        data: {
          photoUploaded: true, photoId: photoIds[0], photoUrl: photoUrls[0],
          photoIds, photoUrls, photoCount: existing.length, updatedAt: new Date()
        }
      }).catch(() => {});
    }

    console.log(`📸 Photo ${Number(index)+1}/${totalExpected} uploaded: ${photoId} (${Math.round(sizeBytes/1024)}KB) user:${req.user.id}`);
    res.json({ success: true, photoId, url: `/api/photos/${photoId}`, sizeKB: Math.round(sizeBytes / 1024), index });
  } catch (e) { console.error('photo upload:', e); res.status(500).json({ error: 'Upload fail hua' }); }
};

export const legacyUploadPhoto = async (req, res) => {
  req.body.index = 0; req.body.totalExpected = 1;
  const { base64, type = 'image/jpeg', consultationId } = req.body;
  if (!base64) return res.status(400).json({ error: 'Image data missing' });
  const photoId = randomBytes(8).toString('hex');
  uploads.set(photoId, { base64, type, userId: req.user.id });
  await prisma.photo.create({
    data: {
      photoId, userId: req.user.id, consultationId: consultationId || null, type, base64, index: 0,
      sizeKB: Math.round((base64.length * 3 / 4) / 1024), url: `/api/photos/${photoId}`
    }
  }).catch(() => {});
  if (consultationId) {
    await prisma.consultation.update({
      where: { id: consultationId },
      data: { photoId, photoUrl: `/api/photos/${photoId}`, photoUploaded: true }
    }).catch(() => {});
  }
  console.log(`📸 (legacy) Photo: ${photoId} user:${req.user.id}`);
  res.json({ success: true, photoId, url: `/api/photos/${photoId}` });
};

export const getPhoto = async (req, res) => {
  try {
    let photo = uploads.get(req.params.photoId);
    if (!photo) {
      const doc = await prisma.photo.findUnique({ where: { photoId: req.params.photoId } });
      if (!doc) return res.status(404).json({ error: 'Photo nahi mili' });
      photo = doc;
      uploads.set(req.params.photoId, photo);
    }
    const rawBase64 = photo.base64.includes(',') ? photo.base64.split(',')[1] : photo.base64;
    const buffer = Buffer.from(rawBase64, 'base64');
    res.set('Content-Type', photo.type || 'image/jpeg');
    res.set('Cache-Control', 'public, max-age=3600');
    res.send(buffer);
  } catch (e) { res.status(500).json({ error: 'Photo serve fail' }); }
};

export const legacyGetPhoto = async (req, res) => {
  req.params.photoId = req.params.id;
  const photo = uploads.get(req.params.id) || await prisma.photo.findUnique({ where: { photoId: req.params.id } }).catch(() => null);
  if (!photo) return res.status(404).json({ error: 'Photo nahi mili' });
  const buf = Buffer.from((photo.base64 || '').split(',')[1] || (photo.base64 || ''), 'base64');
  res.set('Content-Type', photo.type || 'image/jpeg');
  res.send(buf);
};

export const listPhotos = async (req, res) => {
  try {
    const { consultationId } = req.query;
    let photos = [];
    if (consultationId) {
      const consultation = await prisma.consultation.findUnique({ where: { id: consultationId } });
      if (!consultation) return res.status(404).json({ error: 'Consultation nahi mili' });
      const isParticipant = consultation.farmerId === req.user.id || consultation.expertId === req.user.id;
      if (!isParticipant) return res.status(403).json({ error: 'Access denied' });
      const consultPhotoIds = Array.isArray(consultation.photoIds)
        ? consultation.photoIds.filter(Boolean)
        : consultation.photoId ? [consultation.photoId] : [];
      if (consultPhotoIds.length) {
        photos = await prisma.photo.findMany({ where: { photoId: { in: consultPhotoIds } }, orderBy: { index: 'asc' } });
      } else {
        photos = await prisma.photo.findMany({ where: { consultationId }, orderBy: { index: 'asc' } });
      }
    } else {
      photos = await prisma.photo.findMany({ where: { userId: req.user.id }, orderBy: { index: 'asc' } });
    }
    res.json({
      photos: photos.map(p => ({
        photoId: p.photoId, url: p.url, index: p.index,
        sizeKB: p.sizeKB, type: p.type, createdAt: p.createdAt,
        consultationId: p.consultationId,
      })),
      count: photos.length,
    });
  } catch (e) { res.status(500).json({ error: 'Photos nahi mile' }); }
};

export const deletePhoto = async (req, res) => {
  try {
    const photo = await prisma.photo.findFirst({ where: { photoId: req.params.photoId, userId: req.user.id } });
    if (!photo) return res.status(404).json({ error: 'Photo nahi mili ya aapki nahi hai' });
    await prisma.photo.delete({ where: { id: photo.id } }); // Note: Using actual ID, wait photoId is unique, so can use where: { photoId: ... }
    uploads.delete(req.params.photoId);
    if (photo.consultationId) {
      const remaining = await prisma.photo.findMany({ where: { consultationId: photo.consultationId, userId: req.user.id }, orderBy: { index: 'asc' } });
      await prisma.consultation.update({
        where: { id: photo.consultationId },
        data: {
          photoCount: remaining.length, photoUploaded: remaining.length > 0,
          photoIds: remaining.map((p) => p.photoId), photoUrls: remaining.map((p) => p.url),
          photoId: remaining[0]?.photoId || null, photoUrl: remaining[0]?.url || null,
          updatedAt: new Date()
        }
      }).catch(() => {});
    }
    res.json({ success: true, message: 'Photo delete ho gayi' });
  } catch (e) { res.status(500).json({ error: 'Delete fail' }); }
};
