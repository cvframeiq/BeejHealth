import React from 'react';
import {
  API,
  MAX_UPLOAD_DIMENSION,
  MAX_UPLOAD_PAYLOAD_BYTES,
  MAX_UPLOAD_SOURCE_BYTES,
  UPLOAD_JPEG_QUALITY,
} from '../services/api.js';
export function getConsultationContextId() {
  return (
    localStorage.getItem('bh_chat_consult') ||
    localStorage.getItem('bh_view_consult') ||
    localStorage.getItem('bh_latest_consult') ||
    null
  );
}

export function rememberConsultationContext(consultId) {
  if (!consultId) return;
  localStorage.setItem('bh_chat_consult', consultId);
  localStorage.setItem('bh_view_consult', consultId);
}

export function formatChatMessage(msg, locale = 'en-US') {
  const messageType = msg.messageType || (msg.imageUrl || msg.photoId ? 'image' : 'text');
  const imageUrl = msg.imageUrl || (msg.photoId ? `/api/photos/${msg.photoId}` : msg.photoUrl || null);
  return {
    id: msg._id,
    from: msg.senderType,
    text: msg.text || '',
    senderName: msg.senderName,
    messageType,
    imageUrl,
    photoId: msg.photoId || null,
    time: new Date(msg.createdAt).toLocaleTimeString(locale, {hour:'2-digit', minute:'2-digit'}),
  };
}

export function resolveConsultPhotoUrls(consultation) {
  const snapshotUrls = Array.isArray(consultation?.reportSnapshot?.photos)
    ? consultation.reportSnapshot.photos.map((photo) => photo?.url).filter(Boolean)
    : [];
  const urls = Array.isArray(consultation?.photoUrls) && consultation.photoUrls.length
    ? consultation.photoUrls
      : Array.isArray(consultation?.photoIds) && consultation.photoIds.length
      ? consultation.photoIds.map((photoId) => `/api/photos/${photoId}`)
      : consultation?.photoUrl
        ? [consultation.photoUrl]
        : consultation?.photoId
          ? [`/api/photos/${consultation.photoId}`]
          : snapshotUrls;
  return Array.from(new Set([...urls, ...snapshotUrls].filter(Boolean)));
}

export function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (ev) => resolve(ev.target.result);
    reader.onerror = () => reject(new Error('File read fail hua'));
    reader.readAsDataURL(file);
  });
}

export function estimateDataUrlBytes(dataUrl) {
  if (!dataUrl) return 0;
  const commaIndex = dataUrl.indexOf(',');
  const base64 = commaIndex >= 0 ? dataUrl.slice(commaIndex + 1) : dataUrl;
  return Math.ceil((base64.length * 3) / 4);
}

export function inferDataUrlMimeType(dataUrl, fallback = 'image/jpeg') {
  if (typeof dataUrl !== 'string') return fallback;
  const match = dataUrl.match(/^data:([^;,]+)[;,]/i);
  return match?.[1] || fallback;
}

export function isLikelyImageFile(file) {
  const mime = String(file?.type || '').toLowerCase();
  if (mime.startsWith('image/')) return true;
  const name = String(file?.name || '').toLowerCase();
  return /\.(jpg|jpeg|png|webp|heic|heif|gif|bmp|avif)$/i.test(name);
}

export function loadImageFromSrc(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Image load fail hua'));
    img.src = src;
  });
}

export async function optimizeImageSrcToJpeg(src, {
  maxWidth = MAX_UPLOAD_DIMENSION,
  maxHeight = MAX_UPLOAD_DIMENSION,
  quality = UPLOAD_JPEG_QUALITY,
} = {}) {
  const img = await loadImageFromSrc(src);
  const width = img.naturalWidth || img.width;
  const height = img.naturalHeight || img.height;
  if (!width || !height) throw new Error('Image dimensions read nahi hui');
  const scale = Math.min(1, maxWidth / width, maxHeight / height);
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg', quality);
}

export async function fileToUploadDataUrl(file, options = {}) {
  if (!file) throw new Error('File missing');
  const type = String(file.type || '').toLowerCase();
  if (!isLikelyImageFile(file)) throw new Error('Image nahi hai');
  if (file.size > MAX_UPLOAD_SOURCE_BYTES) {
    throw new Error(`Ek photo max ${Math.round(MAX_UPLOAD_SOURCE_BYTES / (1024 * 1024))}MB honi chahiye`);
  }
  const compressibleTypes = new Set(['image/jpeg', 'image/jpg', 'image/png', 'image/webp']);
  if (!compressibleTypes.has(type)) {
    const dataUrl = await fileToDataUrl(file);
    return { dataUrl, contentType: inferDataUrlMimeType(dataUrl, type || 'image/jpeg') };
  }
  const objectUrl = URL.createObjectURL(file);
  try {
    const dataUrl = await optimizeImageSrcToJpeg(objectUrl, options);
    if (estimateDataUrlBytes(dataUrl) > MAX_UPLOAD_PAYLOAD_BYTES) {
      throw new Error('Compressed image abhi bhi bahut badi hai');
    }
    return { dataUrl, contentType: 'image/jpeg' };
  } catch (err) {
    console.warn('Image optimize fail, raw upload fallback:', err.message);
    const dataUrl = await fileToDataUrl(file);
    if (estimateDataUrlBytes(dataUrl) > MAX_UPLOAD_PAYLOAD_BYTES) {
      throw new Error('Image upload ke liye bahut badi hai. Thodi chhoti photo select karein.');
    }
    return { dataUrl, contentType: inferDataUrlMimeType(dataUrl, type || 'image/jpeg') };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

export async function uploadPhotoAsset({ base64, type = 'image/jpeg', consultationId = null, index = 0, totalExpected = 1 }) {
  try {
    return await API.post('/api/photos', {
      base64,
      type,
      consultationId,
      index,
      totalExpected,
    }, 60000);
  } catch (primaryErr) {
    console.warn('Primary /api/photos upload failed, retrying legacy endpoint:', primaryErr.message);
    const legacyRes = await API.post('/api/upload/photo', {
      base64,
      type,
      consultationId,
    }, 60000);
    return {
      ...legacyRes,
      index,
      sizeKB: legacyRes.sizeKB || Math.round((base64.length * 3 / 4) / 1024),
    };
  }
}

export function ChatMessageBody({ msg }) {
  const imageUrl = msg.imageUrl || (msg.photoId ? `/api/photos/${msg.photoId}` : null);
  const hasText = !!msg.text?.trim();
  const hasImage = !!imageUrl;
  if (!hasText && !hasImage) return null;
  return (
    <div style={{display:'flex',flexDirection:'column',gap:hasImage&&hasText?8:0}}>
      {hasImage&&(
        <img
          src={imageUrl}
          alt="Shared"
          style={{width:'100%',maxWidth:260,borderRadius:12,display:'block',objectFit:'cover',boxShadow:'0 8px 22px rgba(12,61,30,.12)'}}
        />
      )}
      {hasText&&<div style={{whiteSpace:'pre-wrap'}}>{msg.text}</div>}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════
   BEEJHEALTH — COMPLETE PRODUCTION APPLICATION
   Farmer Portal + Expert Portal + Full Auth + All Screens
════════════════════════════════════════════════════════════════ */



/* ════════════════════════════════════════════════════════════════
   DATA
════════════════════════════════════════════════════════════════ */
