import express from 'express';
import { getNotifications, readAllNotifications, readNotification, rateExpert, getEarnings, getHealth, getStats, getAdminUsers, resetData, createSupportTicket } from '../controllers/miscController.js';
import { requireAuth, requireExpert } from '../middlewares/authMiddleware.js';

const router = express.Router();

router.get('/notifications', requireAuth, getNotifications);
router.patch('/notifications/read-all', requireAuth, readAllNotifications);
router.patch('/notifications/:id/read', requireAuth, readNotification);

router.post('/experts/:id/rate', requireAuth, rateExpert);
router.get('/earnings', requireAuth, requireExpert, getEarnings);

router.get('/health', getHealth);
router.get('/stats', getStats);
router.get('/admin/users', getAdminUsers);
router.delete('/admin/reset', resetData);
router.post('/support', createSupportTicket);

export default router;
