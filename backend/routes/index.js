import express from 'express';
import authRoutes from './authRoutes.js';
import expertRoutes from './expertRoutes.js';
import consultationRoutes from './consultationRoutes.js';
import photoRoutes from './photoRoutes.js';
import robotRoutes from './robotRoutes.js';
import coconutRoutes from './coconutRoutes.js';
import miscRoutes from './miscRoutes.js';

const router = express.Router();

router.use('/auth', authRoutes);
router.use('/experts', expertRoutes);
router.use('/consultations', consultationRoutes);
router.use('/photos', photoRoutes);
router.use('/robots', robotRoutes);
router.use('/coconut', coconutRoutes);
router.use('/', miscRoutes);

// legacy upload endpoint
import { legacyUploadPhoto, legacyGetPhoto } from '../controllers/photoController.js';
import { requireAuth } from '../middlewares/authMiddleware.js';
router.post('/upload/photo', requireAuth, legacyUploadPhoto);
router.get('/upload/photo/:id', legacyGetPhoto);

export default router;
