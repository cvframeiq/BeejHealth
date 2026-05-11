import express from 'express';
import { checkAIStatus, getQuestions, getQuestionSession, quickScan, analyze, getCoconutConsultations } from '../controllers/coconutController.js';
import { requireAuth } from '../middlewares/authMiddleware.js';

const router = express.Router();

router.get('/ai-status', checkAIStatus);
router.get('/questions', getQuestions);
router.post('/questions/session', getQuestionSession);
router.post('/quick-scan', quickScan);
router.post('/analyze', requireAuth, analyze);
router.get('/consultations', requireAuth, getCoconutConsultations);

export default router;
