import { prisma } from '../db.js';
import { enrichConsultationWithPhotos } from '../utils/helpers.js';
import { setTypingState, getTypingUsers } from '../utils/memoryStore.js';

export const createConsultation = async (req, res) => {
  try {
    const { cropId, cropName, cropEmoji, method, photoUploaded, answers, disease, confidence, severity } = req.body;
    const photoId = req.body.photoId || null;
    const photoUrl = req.body.photoUrl || null;
    const photoIds = Array.isArray(req.body.photoIds) ? req.body.photoIds.filter(Boolean) : (photoId ? [photoId] : []);
    const incomingPhotoUrls = Array.isArray(req.body.photoUrls) ? req.body.photoUrls.filter(Boolean) : (photoUrl ? [photoUrl] : []);
    const resolvedPhotoUrls = incomingPhotoUrls.length ? incomingPhotoUrls : photoIds.map((id) => `/api/photos/${id}`);
    const resolvedPhotoCount = Number(req.body.photoCount) || resolvedPhotoUrls.length || photoIds.length || (photoId ? 1 : 0);

    const experts = await prisma.user.findMany({ where: { type: 'expert', available: true } });
    const expert = experts.length > 0 ? experts[Math.floor(Math.random() * experts.length)] : null;

    const c = await prisma.consultation.create({
      data: {
        farmerId: req.user.id, expertId: expert?.id || null, expertName: expert?.name || 'Pending Assignment',
        cropId: cropId || 'tomato', cropName: cropName || 'Crop', cropEmoji: cropEmoji || '🌱', method: method || 'photo',
        photoUploaded: !!photoUploaded || resolvedPhotoCount > 0, photoId: photoId || photoIds[0] || null,
        photoUrl: photoUrl || resolvedPhotoUrls[0] || (photoIds[0] ? `/api/photos/${photoIds[0]}` : null),
        photoIds, photoUrls: resolvedPhotoUrls, photoCount: resolvedPhotoCount, answers: answers || {},
        disease: disease || 'Analysis Pending', confidence: Number(confidence) || 0, severity: Number(severity) || 1,
        status: expert ? 'expert_assigned' : 'pending', report: null,
      }
    });

    if (photoIds.length) {
      await prisma.photo.updateMany({ where: { photoId: { in: photoIds }, userId: req.user.id }, data: { consultationId: c.id } }).catch(() => {});
    }

    await prisma.notification.create({
      data: {
        userId: req.user.id, type: 'consultation', icon: '🔬', title: `AI Report Ready — ${cropName}`,
        body: `${disease} detected (${confidence}% confidence).${expert ? ` Dr. ${expert.name} assign ho gaya.` : ''}`,
        read: false, consultationId: c.id,
      }
    });

    if (expert) {
      await prisma.notification.create({
        data: {
          userId: expert.id, type: 'new_case', icon: '📋', title: `Naya Case — ${cropName}`,
          body: `${disease} suspected. Immediately review karein.`, read: false, consultationId: c.id,
        }
      });
      await prisma.user.update({ where: { id: expert.id }, data: { totalCases: { increment: 1 } } });
    }

    console.log(`📋 New consultation: ${cropName} | ${disease} | expert: ${expert?.name || 'none'}`);
    res.json({ success: true, consultation: { ...c, _id: c.id } });
  } catch (e) { console.error('consultation:', e); res.status(500).json({ error: 'Consultation save nahi hua' }); }
};

export const getConsultations = async (req, res) => {
  try {
    const q = req.user.type === 'expert' ? { expertId: req.user.id } : { farmerId: req.user.id };
    const list = await prisma.consultation.findMany({ where: q, orderBy: { createdAt: 'desc' } });
    const hydrated = await Promise.all(list.map(c => enrichConsultationWithPhotos(c)));
    const legacyList = hydrated.map(c => ({ ...c, _id: c.id }));
    res.json({ consultations: legacyList });
  } catch (e) { res.status(500).json({ error: 'Consultations nahi mile' }); }
};

export const getConsultationById = async (req, res) => {
  try {
    const c = await prisma.consultation.findUnique({ where: { id: req.params.id } });
    if (!c) return res.status(404).json({ error: 'Consultation nahi mili' });
    const hydrated = await enrichConsultationWithPhotos(c);
    res.json({ consultation: { ...hydrated, _id: hydrated.id } });
  } catch (e) { res.status(500).json({ error: 'Server error' }); }
};

export const updateStatus = async (req, res) => {
  try {
    const { status, report, expertId, expertName } = req.body;
    const upd = { status, updatedAt: new Date() };
    if (report) upd.report = report;
    if (expertId) upd.expertId = expertId;
    if (expertName) upd.expertName = expertName;
    await prisma.consultation.update({ where: { id: req.params.id }, data: upd });

    if (status === 'completed') {
      const c = await prisma.consultation.findUnique({ where: { id: req.params.id } });
      if (c) {
        const expertUser = await prisma.user.findUnique({ where: { id: req.user.id } });
        await prisma.notification.create({
          data: {
            userId: c.farmerId, type: 'report_ready', icon: '✅', title: `Expert Report Ready — ${c.cropName}!`,
            body: `Dr. ${expertUser?.name || 'Expert'} ne aapki ${c.cropName} ki full report bhej di. Abhi dekhein!`,
            read: false, consultationId: c.id,
          }
        });
      }
    }
    res.json({ success: true });
  } catch (e) { res.status(500).json({ error: 'Status update fail' }); }
};

export const updateReport = async (req, res) => {
  try {
    const c = await prisma.consultation.findUnique({ where: { id: req.params.id } });
    if (!c) return res.status(404).json({ error: 'Consultation nahi mili' });
    if (req.user.id !== c.farmerId && req.user.id !== c.expertId) return res.status(403).json({ error: 'Aapko is report par access nahi hai' });

    const { reportSnapshot, reportSummary, report } = req.body;
    if (!reportSnapshot && !reportSummary && !report) return res.status(400).json({ error: 'Report data missing' });

    const upd = { updatedAt: new Date() };
    if (reportSnapshot) {
      let currentSnapshot = c.reportSnapshot;
      if (typeof currentSnapshot === 'string') {
        try { currentSnapshot = JSON.parse(currentSnapshot); } catch (e) {}
      }
      upd.reportSnapshot = { ...(currentSnapshot || {}), ...reportSnapshot };
      if (reportSnapshot.generatedAt) upd.reportGeneratedAt = new Date(reportSnapshot.generatedAt);
      if (reportSnapshot.downloadedAt) upd.reportDownloadedAt = new Date(reportSnapshot.downloadedAt);
    }
    if (reportSummary || report) {
      const summaryText = reportSummary || report;
      upd.aiReportSummary = summaryText;
      if (!c.report) upd.report = summaryText;
    }

    const saved = await prisma.consultation.update({ where: { id: req.params.id }, data: upd });
    res.json({ success: true, consultation: saved });
  } catch (e) { res.status(500).json({ error: 'Report save nahi hua' }); }
};

export const getMessages = async (req, res) => {
  try {
    const msgs = await prisma.chat.findMany({ where: { consultationId: req.params.id }, orderBy: { createdAt: 'asc' } });
    
    // Convert to legacy shape since the frontend might expect imageUrl from the DB
    const legacyMsgs = msgs.map(m => {
      let isImg = m.text.startsWith('[Image:');
      return {
        ...m,
        _id: m.id,
        messageType: isImg ? 'image' : 'text',
        imageUrl: isImg ? m.text.replace('[Image:', '').replace(']', '').trim() : null
      }
    });

    res.json({ messages: legacyMsgs, typing: getTypingUsers(req.params.id, req.user.id) });
  } catch (e) { res.status(500).json({ error: 'Messages nahi mile' }); }
};

export const sendMessage = async (req, res) => {
  try {
    const { text = '', messageType = 'text', imageUrl = null, photoId = null } = req.body;
    const cleanText = String(text || '').trim();
    const resolvedType = imageUrl || photoId ? 'image' : messageType || 'text';
    if (resolvedType !== 'image' && !cleanText) return res.status(400).json({ error: 'Message empty nahi ho sakta' });
    if (resolvedType === 'image' && !imageUrl && !photoId) return res.status(400).json({ error: 'Image attachment missing' });

    let finalImageUrl = imageUrl || (photoId ? `/api/photos/${photoId}` : null);
    let dbText = cleanText;
    if (resolvedType === 'image' && finalImageUrl) {
        dbText = `[Image:${finalImageUrl}] ${cleanText}`;
    }

    const sender = await prisma.user.findUnique({ where: { id: req.user.id } });
    const msg = await prisma.chat.create({
      data: {
        consultationId: req.params.id, senderId: req.user.id, senderName: sender?.name || 'User',
        senderType: req.user.type, text: dbText
      }
    });
    setTypingState(req.params.id, { id: req.user.id, type: req.user.type, name: sender?.name || 'User' }, false);

    const legacyMsg = {
        ...msg,
        _id: msg.id,
        messageType: resolvedType,
        imageUrl: finalImageUrl
    };

    const c = await prisma.consultation.findUnique({ where: { id: req.params.id } });
    const toId = c && (req.user.type === 'expert' ? c.farmerId : c.expertId);
    if (toId) {
      await prisma.notification.create({
        data: {
          userId: toId, type: 'message', icon: '💬', title: `${sender?.name || 'User'} ne ${resolvedType === 'image' ? 'image' : 'message'} bheja`,
          body: resolvedType === 'image' ? `📷 ${sender?.name || 'User'} ne image share ki` : cleanText.slice(0, 100),
          read: false, consultationId: req.params.id,
        }
      });
    }
    res.json({ success: true, message: legacyMsg });
  } catch (e) { res.status(500).json({ error: 'Message send nahi hua' }); }
};

export const updateTyping = async (req, res) => {
  try {
    const c = await prisma.consultation.findUnique({ where: { id: req.params.id } });
    if (!c) return res.status(404).json({ error: 'Consultation nahi mili' });
    const sender = await prisma.user.findUnique({ where: { id: req.user.id } });
    setTypingState(req.params.id, { id: req.user.id, type: req.user.type, name: sender?.name || 'User' }, !!req.body.isTyping);
    res.json({ success: true, typing: getTypingUsers(req.params.id, req.user.id) });
  } catch (e) { res.status(500).json({ error: 'Typing update fail' }); }
};
