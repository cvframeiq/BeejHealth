import { prisma } from '../db.js';

async function ensureRobots(userId) {
  const existing = await prisma.robot.findMany({ where: { ownerId: userId } });
  if (existing.length > 0) return existing;
  const defaults = [
    { robotId:'R01', name:'DroneBot Alpha', type:'Drone', model:'DJI Agras T40', status:'online', battery:87, signal:94, field:'Field 1', task:'Standby', emoji:'🚁', sprayArea:0, flights:124, totalArea:'2.1 Acres', ownerId:userId },
    { robotId:'R02', name:'GroundBot Beta', type:'Ground', model:'TartanSense TG-1', status:'online', battery:62, signal:88, field:'Field 2', task:'Standby', emoji:'🤖', sprayArea:0, flights:0, totalArea:'1.5 Acres', ownerId:userId },
    { robotId:'R03', name:'DroneBot Gamma', type:'Drone', model:'ideaForge RYNO', status:'offline', battery:12, signal:0, field:'Charging', task:'Charging', emoji:'🚁', sprayArea:0, flights:89, totalArea:'—', ownerId:userId },
    { robotId:'R04', name:'SensorBot Delta', type:'Sensor', model:'Custom IoT v2', status:'online', battery:91, signal:99, field:'All Fields', task:'Monitoring', emoji:'📡', sprayArea:0, flights:0, totalArea:'4.5 Acres', ownerId:userId },
  ];
  await Promise.all(defaults.map(r => prisma.robot.upsert({ where: { robotId: r.robotId }, update: {}, create: r })));
  return await prisma.robot.findMany({ where: { ownerId: userId } });
}

export const getRobots = async (req, res) => {
  try {
    const robots = await ensureRobots(req.user.id);
    const live = robots.map(r => ({
      ...r,
      battery: r.status === 'online' ? Math.max(10, r.battery - Math.floor(Math.random() * 2)) : r.battery,
      lastSeen: r.status === 'online' ? 'Just now' : r.lastSeen || '2 hrs ago',
    }));
    res.json({ robots: live, total: live.length, online: live.filter(r=>r.status==='online').length, offline: live.filter(r=>r.status==='offline').length });
  } catch(e) { console.error(e); res.status(500).json({ error: 'Robots nahi mile' }); }
};

export const getRobotById = async (req, res) => {
  try {
    let robot = await prisma.robot.findFirst({ where: { robotId: req.params.robotId, ownerId: req.user.id } });
    if (!robot) {
      const all = await ensureRobots(req.user.id);
      robot = all.find(r => r.robotId === req.params.robotId);
    }
    if (!robot) return res.status(404).json({ error: 'Robot nahi mila' });
    res.json({ robot });
  } catch(e) { res.status(500).json({ error: 'Server error' }); }
};

export const updateRobot = async (req, res) => {
  try {
    await ensureRobots(req.user.id);
    const { status, field, task, battery } = req.body;
    const upd = { updatedAt: new Date() };
    if (status) upd.status = status;
    if (field) upd.field = field;
    if (task) upd.task = task;
    if (battery !== undefined) upd.battery = battery;
    
    await prisma.robot.updateMany({ where: { robotId: req.params.robotId, ownerId: req.user.id }, data: upd });
    const robot = await prisma.robot.findFirst({ where: { robotId: req.params.robotId, ownerId: req.user.id } });
    await prisma.robotLog.create({ data: { robotId: req.params.robotId, ownerId: req.user.id, event: `Status changed to ${status || 'updated'}`, level: 'info' } });
    res.json({ success: true, robot });
  } catch(e) { res.status(500).json({ error: 'Robot update fail' }); }
};

export const commandRobot = async (req, res) => {
  try {
    const { command, params } = req.body;
    if (!command) return res.status(400).json({ error: 'Command required' });
    const validCommands = ['start', 'stop', 'pause', 'resume', 'return_home', 'emergency_stop', 'move', 'spray_start', 'spray_stop', 'take_photo', 'wake_up'];
    if (!validCommands.includes(command)) return res.status(400).json({ error: `Unknown command. Valid: ${validCommands.join(', ')}` });

    const statusMap = { start: 'busy', stop: 'online', pause: 'online', resume: 'busy', return_home: 'busy', emergency_stop: 'online', wake_up: 'online', spray_start: 'busy', spray_stop: 'online' };
    if (statusMap[command]) {
      await prisma.robot.updateMany({ where: { robotId: req.params.robotId, ownerId: req.user.id }, data: { status: statusMap[command], task: command.replace(/_/g,' '), updatedAt: new Date() } });
    }
    await prisma.robotLog.create({ data: { robotId: req.params.robotId, ownerId: req.user.id, event: `Command: ${command}${params ? ' ' + JSON.stringify(params) : ''}`, level: command === 'emergency_stop' ? 'warning' : 'info' } });
    console.log(`🤖 Robot ${req.params.robotId} ← ${command}`);
    res.json({ success: true, command, acknowledged: true, timestamp: new Date().toISOString() });
  } catch(e) { res.status(500).json({ error: 'Command fail' }); }
};

export const getRobotLogs = async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 20;
    let logs = await prisma.robotLog.findMany({ where: { robotId: req.params.robotId, ownerId: req.user.id }, orderBy: { createdAt: 'desc' }, take: limit });

    if (logs.length === 0) {
      const now = Date.now();
      logs = [
        { event: 'System online — all sensors nominal', level: 'info', createdAt: new Date(now - 1000*60*2).toISOString() },
        { event: 'GPS lock acquired — 18.59°N 73.74°E', level: 'info', createdAt: new Date(now - 1000*60*8).toISOString() },
        { event: 'Battery charge complete — 87%', level: 'info', createdAt: new Date(now - 1000*60*18).toISOString() },
        { event: 'Field 1 spray mission complete', level: 'info', createdAt: new Date(now - 1000*60*42).toISOString() },
        { event: 'Low battery warning — 18%', level: 'warning', createdAt: new Date(now - 1000*60*70).toISOString() },
      ];
    }
    res.json({ logs, count: logs.length });
  } catch(e) { res.status(500).json({ error: 'Logs nahi mile' }); }
};

export const getRobotCamera = async (req, res) => {
  try {
    await ensureRobots(req.user.id);
    const robot = await prisma.robot.findFirst({ where: { robotId: req.params.robotId, ownerId: req.user.id } });
    if (!robot) return res.status(404).json({ error: 'Robot nahi mila' });

    const cameras = {
      Drone: [{ id:'front', name:'Front Cam', type:'RGB', resolution:'4K/30fps', bitrate:'12 Mbps', latency:'~220ms' }, { id:'bottom', name:'Bottom Cam', type:'RGB+IR', resolution:'4K/30fps', bitrate:'10 Mbps', latency:'~220ms' }],
      Ground: [{ id:'front', name:'Front Cam', type:'RGB', resolution:'1080p/30fps', bitrate:'8 Mbps', latency:'~180ms' }, { id:'soil', name:'Soil Cam', type:'Multispect', resolution:'720p/10fps', bitrate:'4 Mbps', latency:'~250ms' }],
      Sensor: [{ id:'wide', name:'Wide Angle', type:'RGB', resolution:'1080p/15fps', bitrate:'6 Mbps', latency:'~300ms' }],
    };
    const cams = cameras[robot.type] || cameras['Drone'];

    const detections = robot.status !== 'offline' ? [
      { x:35, y:25, w:15, h:18, label:'Early Blight', conf:94, color:'#ff4444', action:'Spray needed' },
      { x:62, y:48, w:12, h:14, label:'Healthy Leaf', conf:98, color:'#00ff9d', action:'OK' },
      { x:18, y:60, w:10, h:12, label:'Pest Damage', conf:79, color:'#ffd700', action:'Monitor' },
    ] : [];

    res.json({
      robotId: robot.robotId, robotName: robot.name, isLive: robot.status !== 'offline', cameras: cams, detections,
      gps: { lat: 18.5913 + Math.random()*0.001, lng: 73.7416 + Math.random()*0.001 },
      altitude: robot.type === 'Drone' ? `${11 + Math.floor(Math.random()*3)}m` : '0m',
      speed: robot.type === 'Drone' ? `${3 + (Math.random()*1.5).toFixed(1)}m/s` : `${0.5 + (Math.random()*0.8).toFixed(1)}m/s`,
    });
  } catch(e) { res.status(500).json({ error: 'Camera info nahi mili' }); }
};

export const takeSnapshot = async (req, res) => {
  try {
    const { base64, cameraId } = req.body;
    const { randomBytes } = await import('crypto');
    const snapId = randomBytes(6).toString('hex');
    const { uploads } = await import('../utils/memoryStore.js');
    if (base64) uploads.set('snap_' + snapId, { base64, type:'image/jpeg', userId: req.user.id, ts: Date.now() });
    await prisma.robotLog.create({ data: { robotId: req.params.robotId, ownerId: req.user.id, event: `Snapshot taken — cam:${cameraId||'front'} id:${snapId}`, level: 'info' } });
    res.json({ success: true, snapId, url: base64 ? `/api/upload/photo/snap_${snapId}` : null });
  } catch(e) { res.status(500).json({ error: 'Snapshot fail' }); }
};

export const getAllCameras = async (req, res) => {
  try {
    const robots = await ensureRobots(req.user.id);
    const feeds = robots.map(r => ({
      robotId: r.robotId, robotName: r.name, type: r.type, emoji: r.emoji, field: r.field, status: r.status, isLive: r.status !== 'offline',
      primaryCam: r.type === 'Drone' ? 'Front 4K' : r.type === 'Ground' ? 'Front 1080p' : 'Wide 1080p',
    }));
    res.json({ feeds, total: feeds.length, live: feeds.filter(f=>f.isLive).length });
  } catch(e) { res.status(500).json({ error: 'Camera feeds nahi mile' }); }
};

export const getSprayJobs = async (req, res) => {
  try {
    let jobs = await prisma.sprayJob.findMany({ where: { ownerId: req.user.id }, orderBy: { scheduledAt: 'asc' } });
    if (jobs.length === 0) {
      const now = new Date();
      const seeds = [
        { jobId:'SJ001', robotId:'R01', field:'Field 1', crop:'Tamatar', chemical:'Mancozeb 75% WP', dose:2.5, area:2.0, status:'scheduled', scheduledAt: new Date(now.getTime()+3600000), priority:'high', disease:'Early Blight', ownerId:req.user.id },
        { jobId:'SJ002', robotId:'R01', field:'Field 2', crop:'Gehun', chemical:'Copper Oxychloride', dose:3.0, area:1.5, status:'scheduled', scheduledAt: new Date(now.getTime()+86400000), priority:'med', disease:'Leaf Rust (preventive)', ownerId:req.user.id },
        { jobId:'SJ003', robotId:'R02', field:'Field 3', crop:'Aalu', chemical:'Mancozeb 75% WP', dose:2.5, area:1.0, status:'completed', scheduledAt: new Date(now.getTime()-86400000), priority:'low', disease:'Late Blight (prev)', ownerId:req.user.id },
      ];
      const promises = seeds.map(s => prisma.sprayJob.upsert({ where: { jobId: s.jobId }, update: {}, create: s }));
      await Promise.all(promises);
      jobs = await prisma.sprayJob.findMany({ where: { ownerId: req.user.id }, orderBy: { scheduledAt: 'asc' } });
    }
    res.json({ jobs, total: jobs.length, pending: jobs.filter(j=>j.status==='scheduled').length, completed: jobs.filter(j=>j.status==='completed').length });
  } catch(e) { res.status(500).json({ error: 'Spray jobs nahi mile' }); }
};

export const createSprayJob = async (req, res) => {
  try {
    const { robotId, field, crop, chemical, dose, area, scheduledAt, priority, disease } = req.body;
    if (!field || !chemical || !scheduledAt) return res.status(400).json({ error: 'field, chemical, scheduledAt required' });
    const { randomBytes } = await import('crypto');
    const job = await prisma.sprayJob.create({
      data: {
        jobId: 'SJ' + randomBytes(3).toString('hex').toUpperCase(), robotId: robotId || 'R01', field, crop: crop || 'Fasal',
        chemical, dose: Number(dose) || 2.5, area: Number(area) || 1.0, status: 'scheduled', scheduledAt: new Date(scheduledAt), priority: priority || 'med',
        disease: disease || 'Preventive', ownerId: req.user.id,
      }
    });
    await prisma.robotLog.create({ data: { robotId: robotId || 'R01', ownerId: req.user.id, event: `Spray job scheduled — ${field} | ${chemical} | ${new Date(scheduledAt).toLocaleString('en-IN')}`, level: 'info' } });
    res.json({ success: true, job });
  } catch(e) { res.status(500).json({ error: 'Spray job create fail' }); }
};

export const updateSprayJob = async (req, res) => {
  try {
    const { status } = req.body;
    await prisma.sprayJob.updateMany({ where: { jobId: req.params.jobId, ownerId: req.user.id }, data: { status, updatedAt: new Date() } });
    const job = await prisma.sprayJob.findUnique({ where: { jobId: req.params.jobId } });
    res.json({ success: true, job });
  } catch(e) { res.status(500).json({ error: 'Status update fail' }); }
};

export const deleteSprayJob = async (req, res) => {
  try {
    await prisma.sprayJob.deleteMany({ where: { jobId: req.params.jobId, ownerId: req.user.id } });
    res.json({ success: true, message: 'Spray job cancel ho gaya' });
  } catch(e) { res.status(500).json({ error: 'Cancel fail' }); }
};

export const getLocation = async (req, res) => {
  try {
    const robot = await prisma.robot.findFirst({ where: { robotId: req.params.robotId, ownerId: req.user.id } });
    if (!robot) return res.status(404).json({ error: 'Robot nahi mila' });
    const isLive = robot.status !== 'offline';
    const baseLat = 18.5913; const baseLng = 73.7416;
    res.json({
      robotId: req.params.robotId, isLive,
      gps: isLive ? { lat: baseLat + Math.random() * 0.003, lng: baseLng + Math.random() * 0.003, accuracy: '±2m', altitude: robot.type === 'Drone' ? `${11 + Math.floor(Math.random()*4)}m` : '0m', heading: Math.floor(Math.random() * 360), speed: isLive ? `${(Math.random()*4).toFixed(1)}m/s` : '0m/s' } : null,
      field: robot.field,
      waypoints: isLive ? [{ lat: baseLat + 0.001, lng: baseLng + 0.001, label: 'WP1', done: true }, { lat: baseLat + 0.002, lng: baseLng + 0.002, label: 'WP2', done: true }, { lat: baseLat + 0.003, lng: baseLng + 0.003, label: 'WP3', done: false }, { lat: baseLat + 0.001, lng: baseLng + 0.003, label: 'WP4', done: false }] : [],
    });
  } catch(e) { res.status(500).json({ error: 'Location nahi mili' }); }
};

export const navigateRobot = async (req, res) => {
  try {
    const { waypoints, mode, field } = req.body;
    if (!mode) return res.status(400).json({ error: 'mode required (auto/manual/return)' });
    await prisma.robot.updateMany({ where: { robotId: req.params.robotId, ownerId: req.user.id }, data: { status: 'busy', task: `Navigation: ${mode}`, field: field || 'Field', updatedAt: new Date() } });
    await prisma.robotLog.create({ data: { robotId: req.params.robotId, ownerId: req.user.id, event: `Mission started — mode:${mode} waypoints:${(waypoints||[]).length}`, level: 'info' } });
    res.json({ success: true, mode, waypointCount: (waypoints||[]).length, eta: '~12 min' });
  } catch(e) { res.status(500).json({ error: 'Navigation fail' }); }
};

export const getAnalyticsSummary = async (req, res) => {
  try {
    const robots = await ensureRobots(req.user.id);
    const jobs = await prisma.sprayJob.findMany({ where: { ownerId: req.user.id } });
    const logs = await prisma.robotLog.findMany({ where: { ownerId: req.user.id } });
    const completed = jobs.filter(j => j.status === 'completed');
    const totalArea = completed.reduce((s, j) => s + (j.area || 0), 0);
    const totalDose = completed.reduce((s, j) => s + (j.dose || 0), 0);
    res.json({
      fleetSize: robots.length, onlineNow: robots.filter(r => r.status === 'online').length, missionsDone: completed.length,
      areaCovered: `${totalArea.toFixed(1)} Acres`, sprayVolume: `${(totalDose * totalArea).toFixed(0)}L`,
      totalFlights: robots.reduce((s, r) => s + (r.flights || 0), 0), avgBattery: Math.round(robots.reduce((s, r) => s + r.battery, 0) / (robots.length || 1)),
      logsToday: logs.filter(l => new Date(l.createdAt).toDateString() === new Date().toDateString()).length,
      weeklyData: [ { day:'Mon', area: 1.2, spray: 180 }, { day:'Tue', area: 0.8, spray: 120 }, { day:'Wed', area: 2.1, spray: 315 }, { day:'Thu', area: 1.5, spray: 225 }, { day:'Fri', area: 1.8, spray: 270 }, { day:'Sat', area: 0.6, spray: 90 }, { day:'Sun', area: 0.3, spray: 45 } ],
    });
  } catch(e) { res.status(500).json({ error: 'Analytics nahi mile' }); }
};

export const getMaintenanceInfo = async (req, res) => {
  try {
    await ensureRobots(req.user.id);
    const robot = await prisma.robot.findFirst({ where: { robotId: req.params.robotId, ownerId: req.user.id } });
    if (!robot) return res.status(404).json({ error: 'Robot nahi mila' });
    const flightHours = Math.round((robot.flights || 0) * 0.45);
    res.json({
      robotId: req.params.robotId, name: robot.name,
      battery: { level: robot.battery, health: robot.battery > 50 ? 'Good' : robot.battery > 20 ? 'Fair' : 'Critical', cycles: robot.flights || 0, nextSwap: robot.battery < 20 ? 'Immediate' : robot.battery < 50 ? 'This week' : 'Next month' },
      motors: { status: robot.status !== 'offline' ? 'Operational' : 'Unknown', lastCheck: '3 days ago', hoursUsed: flightHours },
      sprayer: { tankLevel: robot.type === 'Drone' ? 75 : 0, nozzleWear: robot.flights > 100 ? 'Replace soon' : 'Good', lastClean: '2 days ago' },
      alerts: [ ...(robot.battery < 20 ? [{ level:'critical', msg:'Battery critical — charge immediately' }] : []), ...(robot.flights > 120 ? [{ level:'warning', msg:'Motor inspection due — 124 flights logged' }] : []), ...(robot.signal < 30 ? [{ level:'warning', msg:'Weak signal — check antenna' }] : []) ],
      nextService: '7 Nov 2026',
    });
  } catch(e) { res.status(500).json({ error: 'Maintenance data nahi mila' }); }
};
