import express from 'express';
import { getRobots, getRobotById, updateRobot, commandRobot, getRobotLogs, getRobotCamera, takeSnapshot, getAllCameras, getSprayJobs, createSprayJob, updateSprayJob, deleteSprayJob, getLocation, navigateRobot, getAnalyticsSummary, getMaintenanceInfo } from '../controllers/robotController.js';
import { requireAuth } from '../middlewares/authMiddleware.js';

const router = express.Router();

// Camera
router.get('/camera/all', requireAuth, getAllCameras);

// Spray Jobs
router.get('/spray-jobs', requireAuth, getSprayJobs);
router.post('/spray-jobs', requireAuth, createSprayJob);
router.patch('/spray-jobs/:jobId', requireAuth, updateSprayJob);
router.delete('/spray-jobs/:jobId', requireAuth, deleteSprayJob);

// Analytics
router.get('/analytics/summary', requireAuth, getAnalyticsSummary);

// Robots
router.get('/', requireAuth, getRobots);
router.get('/:robotId', requireAuth, getRobotById);
router.patch('/:robotId', requireAuth, updateRobot);
router.post('/:robotId/command', requireAuth, commandRobot);
router.get('/:robotId/logs', requireAuth, getRobotLogs);
router.get('/:robotId/camera', requireAuth, getRobotCamera);
router.post('/:robotId/camera/snapshot', requireAuth, takeSnapshot);
router.get('/:robotId/location', requireAuth, getLocation);
router.post('/:robotId/navigate', requireAuth, navigateRobot);
router.get('/:robotId/maintenance', requireAuth, getMaintenanceInfo);

export default router;
