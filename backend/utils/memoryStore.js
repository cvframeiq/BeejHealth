export const uploads = new Map(); // in-memory cache for fast retrieval
export const chatTypingState = new Map();
export const TYPING_TTL_MS = 3500;
export const MAX_PHOTO_UPLOAD_BYTES = 25 * 1024 * 1024;

export function pruneTypingRoom(room) {
  const now = Date.now();
  for (const [userId, info] of room.entries()) {
    if (!info || now - info.updatedAt > TYPING_TTL_MS) room.delete(userId);
  }
}

export function getTypingUsers(consultationId, viewerId = null) {
  const room = chatTypingState.get(consultationId);
  if (!room) return [];
  pruneTypingRoom(room);
  const typing = [];
  for (const [userId, info] of room.entries()) {
    if (userId === viewerId) continue;
    typing.push({
      userId,
      senderType: info.senderType || 'farmer',
      senderName: info.senderName || 'User',
      updatedAt: info.updatedAt,
    });
  }
  if (room.size === 0) chatTypingState.delete(consultationId);
  return typing;
}

export function setTypingState(consultationId, user, isTyping) {
  if (!consultationId || !user?.id) return;
  let room = chatTypingState.get(consultationId);
  if (!room) {
    room = new Map();
    chatTypingState.set(consultationId, room);
  }
  if (isTyping) {
    room.set(user.id, {
      senderType: user.type,
      senderName: user.name || 'User',
      updatedAt: Date.now(),
    });
  } else {
    room.delete(user.id);
  }
  pruneTypingRoom(room);
  if (room.size === 0) chatTypingState.delete(consultationId);
}
