import express from 'express';
import { sendOtp, login, register, getMe, updateProfile, updatePassword } from '../controllers/authController.js';
import { requireAuth } from '../middlewares/authMiddleware.js';

const router = express.Router();

router.post('/send-otp', sendOtp);
router.post('/login', login);
router.post('/register', register);
router.get('/me', requireAuth, getMe);
router.patch('/profile', requireAuth, updateProfile);
router.patch('/password', requireAuth, updatePassword);

export default router;
