import express from 'express';
import { createConsultation, getConsultations, getConsultationById, updateStatus, updateReport, getMessages, sendMessage, updateTyping } from '../controllers/consultationController.js';
import { requireAuth } from '../middlewares/authMiddleware.js';

const router = express.Router();

router.post('/', requireAuth, createConsultation);
router.get('/', requireAuth, getConsultations);
router.get('/:id', requireAuth, getConsultationById);
router.patch('/:id/status', requireAuth, updateStatus);
router.patch('/:id/report', requireAuth, updateReport);
router.get('/:id/messages', requireAuth, getMessages);
router.post('/:id/messages', requireAuth, sendMessage);
router.post('/:id/typing', requireAuth, updateTyping);

export default router;
