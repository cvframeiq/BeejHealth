export const PORT       = process.env.PORT || 3000;
export const JWT_SECRET = process.env.JWT_SECRET || 'bh-jwt-secret-change-in-production-2024';
export const FRONTEND   = (process.env.FRONTEND_URL || 'http://localhost:5173').split(',');
export const AI_SERVER  = process.env.AI_SERVER_URL || 'http://localhost:8000';
