import express from 'express';
import { getExperts, getExpertById, updateAvailability } from '../controllers/expertController.js';
import { requireAuth, requireExpert } from '../middlewares/authMiddleware.js';

const router = express.Router();

router.get('/', getExperts);
router.get('/:id', getExpertById);
router.patch('/availability', requireAuth, requireExpert, updateAvailability);

export default router;
