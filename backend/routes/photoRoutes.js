import express from 'express';
import { uploadPhoto, getPhoto, listPhotos, deletePhoto } from '../controllers/photoController.js';
import { requireAuth } from '../middlewares/authMiddleware.js';

const router = express.Router();

router.post('/', requireAuth, uploadPhoto);
router.get('/', requireAuth, listPhotos);
router.get('/:photoId', getPhoto);
router.delete('/:photoId', requireAuth, deletePhoto);

export default router;
